"""Overlay rendering helpers for RendererV3.

This mixin keeps the low-level pygame/cv2 overlay, perspective and occlusion
helpers out of synthgen.py while preserving the RendererV3 method names.
"""

import cv2
import numpy as np


class RendererOverlayMixin:
    def _get_cached_kernel(self, k: int):
        k = int(max(1, k))
        ker = self._kernel_cache.get(k, None) if hasattr(self, "_kernel_cache") else None
        if ker is None:
            ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            if not hasattr(self, "_kernel_cache"):
                self._kernel_cache = {}
            self._kernel_cache[k] = ker
        return ker

    def _pygame_init_once(self):
        try:
            import pygame
            if not getattr(self, "_pygame_inited", False):
                if not pygame.get_init():
                    pygame.init()
                self._pygame_inited = True
        except Exception:
            pass

    def _overlay_get_surface(self, Wc: int, Hc: int):
        """
        Переиспользуем pygame.Surface чтобы меньше аллокаций.
        """
        import pygame
        key = (int(Wc), int(Hc))
        cache = getattr(self, "_overlay_surf_cache", None)
        if cache is None:
            cache = {}
            self._overlay_surf_cache = cache
        surf = cache.get(key, None)
        if surf is None:
            surf = pygame.Surface((key[0], key[1]), flags=pygame.SRCALPHA)
            cache[key] = surf
        surf.fill((0, 0, 0, 0))
        return surf


    def _compute_budgets(self, nregions: int, target_blocks: int):
        """
        Динамически подбираем бюджеты:
        - мало регионов/блоков -> агрессивно режем попытки;
        - много регионов -> даём чуть больше воздуха.
        """
        k = max(1, int(nregions))
        g = max(3, int(self.global_attempt_budget_base * (1.0 + 0.15*(k-1)) * (0.6 + 0.2*target_blocks)))
        pr = max(2, int(self.per_region_attempt_cap_base * (0.8 + 0.1*min(k,5))))
        ms = self.max_shrink_trials_base
        return g, pr, ms
    

    def _wrap_deg_pm180(self, a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0


    def _edge_len(self, p, q) -> float:
        import math
        v0 = float(q[0] - p[0])
        v1 = float(q[1] - p[1])
        return float(math.hypot(v0, v1))


    def _poly_area(self, q) -> float:
        import numpy as np
        q = np.asarray(q, dtype=np.float32).reshape(-1, 2)
        x = q[:, 0].astype(np.float64)
        y = q[:, 1].astype(np.float64)
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


    def _persp_strength_from_quad(self, q):
        """
        Возвращает:
        strength: "насколько перспективно" (0 ~ аффинно)
        affine_like: bool
        met: детали (w_top, w_bot, h_lft, h_rgt, w_ratio, h_ratio)
        """
        import math
        q = q.reshape(4, 2)

        w_top = self._edge_len(q[0], q[1])
        w_bot = self._edge_len(q[3], q[2])
        h_lft = self._edge_len(q[0], q[3])
        h_rgt = self._edge_len(q[1], q[2])

        w_ratio = w_top / max(1e-6, w_bot)
        h_ratio = h_lft / max(1e-6, h_rgt)

        strength = max(
            abs(math.log(max(1e-6, w_ratio))),
            abs(math.log(max(1e-6, h_ratio))),
        )
        affine_like = (strength < 0.02)
        return float(strength), bool(affine_like), (w_top, w_bot, h_lft, h_rgt, w_ratio, h_ratio)


    def _scale_edge(self, dst_quad, i0: int, i1: int, scale: float):
        import numpy as np
        q = np.asarray(dst_quad, dtype=np.float32).reshape(4, 2).copy()
        p0 = q[i0].copy()
        p1 = q[i1].copy()
        m = 0.5 * (p0 + p1)
        k = float(scale)
        k = max(0.05, min(2.50, k))
        q[i0] = m + (p0 - m) * k
        q[i1] = m + (p1 - m) * k
        return q


    def _apply_persp_boost(
        self,
        dst_quad,
        boost_factor: float,
        min_ratio_eff: float,
        *,
        prefer_axis="w",
        far_by_y=True,
        expand_near=True,
    ):
        """
        Усиливает перспективу у quad: сжимает "дальнюю" грань и чуть расширяет "ближнюю".
        """
        import numpy as np
        import math

        q = np.asarray(dst_quad, dtype=np.float32).reshape(4, 2).copy()

        w_top = self._edge_len(q[0], q[1])
        w_bot = self._edge_len(q[3], q[2])
        h_lft = self._edge_len(q[0], q[3])
        h_rgt = self._edge_len(q[1], q[2])

        w_ratio = w_top / max(1e-6, w_bot)
        h_ratio = h_lft / max(1e-6, h_rgt)

        if prefer_axis == "auto":
            dw = abs(math.log(max(1e-6, min(w_ratio, 1.0 / max(w_ratio, 1e-6)))))
            dh = abs(math.log(max(1e-6, min(h_ratio, 1.0 / max(h_ratio, 1e-6)))))
            use_w = (dw >= dh)
        elif prefer_axis == "h":
            use_w = False
        else:
            use_w = True

        if use_w:
            mid_y_top = 0.5 * (float(q[0, 1]) + float(q[1, 1]))
            mid_y_bot = 0.5 * (float(q[3, 1]) + float(q[2, 1]))

            if far_by_y:
                far_is_top = (mid_y_top <= mid_y_bot)
            else:
                far_is_top = (w_top <= w_bot)

            if far_is_top:
                far_i0, far_i1 = 0, 1
                near_i0, near_i1 = 3, 2
                len_far, len_near = w_top, w_bot
            else:
                far_i0, far_i1 = 3, 2
                near_i0, near_i1 = 0, 1
                len_far, len_near = w_bot, w_top
        else:
            mid_x_left = 0.5 * (float(q[0, 0]) + float(q[3, 0]))
            mid_x_right = 0.5 * (float(q[1, 0]) + float(q[2, 0]))
            cx = 0.5 * (mid_x_left + mid_x_right)
            far_is_left = (abs(mid_x_left - cx) <= abs(mid_x_right - cx))

            if far_is_left:
                far_i0, far_i1 = 0, 3
                near_i0, near_i1 = 1, 2
                len_far, len_near = h_lft, h_rgt
            else:
                far_i0, far_i1 = 1, 2
                near_i0, near_i1 = 0, 3
                len_far, len_near = h_rgt, h_lft

        ratio = float(len_far / max(1e-6, len_near))  # far/near

        if ratio >= 0.98:
            ratio_new = 1.0 / (1.0 + 0.55 * float(boost_factor))
        else:
            ratio_new = ratio ** float(boost_factor)

        ratio_new = max(float(min_ratio_eff), min(0.98, float(ratio_new)))

        s_far = ratio_new / max(1e-6, ratio)
        s_far = min(1.0, float(s_far))
        q = self._scale_edge(q, far_i0, far_i1, scale=s_far)

        if expand_near:
            near_expand = min(0.35, 0.06 * float(boost_factor))
            s_near = 1.0 + near_expand
            q = self._scale_edge(q, near_i0, near_i1, scale=s_near)

        return q


    def _is_sky_like_quad_geom(self, q, H_img: int, W_img: int):
        """
        Геометрическая эвристика "похоже на небо".
        """
        import numpy as np
        q = np.asarray(q, dtype=np.float32).reshape(4, 2)

        x0 = float(np.min(q[:, 0])); x1 = float(np.max(q[:, 0]))
        y0 = float(np.min(q[:, 1])); y1 = float(np.max(q[:, 1]))
        bw = (x1 - x0) / max(1.0, float(W_img))
        bh = (y1 - y0) / max(1.0, float(H_img))
        yc = ((y0 + y1) * 0.5) / max(1.0, float(H_img))
        area = float(self._poly_area(q)) / max(1.0, float(H_img * W_img))

        area_thr = float(getattr(self, "overlay_sky_area_thr", 0.18))
        y_thr    = float(getattr(self, "overlay_sky_y_thr", 0.40))
        w_thr    = float(getattr(self, "overlay_sky_w_thr", 0.80))
        h_thr    = float(getattr(self, "overlay_sky_h_thr", 0.25))

        if area < 0.01:
            return False, (area, yc, bw, bh)

        sky_like = (area >= area_thr) and (yc <= y_thr) and (bw >= w_thr) and (bh >= h_thr)
        return bool(sky_like), (area, yc, bw, bh)


    def _angle_in_sector_deg(self, a: float, a0: float, a1: float) -> bool:
        a = float(a) % 360.0
        a0 = float(a0) % 360.0
        a1 = float(a1) % 360.0
        if a0 <= a1:
            return (a0 <= a <= a1)
        return (a >= a0) or (a <= a1)


    def _sky_sector_vote(self, q, H_img: int, W_img: int):
        """
        Доп. эвристика: quad попадает в верхнюю часть кадра/сектор "неба".
        """
        import numpy as np
        import math

        enabled = bool(getattr(self, "overlay_sky_sector_enable", True))
        if not enabled:
            return False, None

        a0 = float(getattr(self, "overlay_sky_sector_deg_min", 45.0))
        a1 = float(getattr(self, "overlay_sky_sector_deg_max", 135.0))
        votes_thr = int(getattr(self, "overlay_sky_sector_votes_thr", 5))
        min_r = float(getattr(self, "overlay_sky_sector_min_r", 0.08))
        require_above = bool(getattr(self, "overlay_sky_sector_require_above_center", True))

        q = np.asarray(q, dtype=np.float32).reshape(4, 2)
        cx0 = 0.5 * float(W_img)
        cy0 = 0.5 * float(H_img)
        Rmax = math.hypot(cx0, cy0) + 1e-6

        pts = []
        pts.extend([q[0], q[1], q[2], q[3]])
        pts.extend([
            0.5*(q[0]+q[1]),
            0.5*(q[1]+q[2]),
            0.5*(q[2]+q[3]),
            0.5*(q[3]+q[0]),
        ])
        pts.append(np.mean(q, axis=0))

        votes = 0
        for p in pts:
            dx = float(p[0] - cx0)
            dy = float(p[1] - cy0)
            r = math.hypot(dx, dy) / Rmax
            ang = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0

            ok_r = (r >= min_r)
            ok_above = (dy < 0.0) if require_above else True
            ok_ang = self._angle_in_sector_deg(ang, a0, a1)

            votes += 1 if (ok_r and ok_above and ok_ang) else 0

        is_sky = (votes >= votes_thr)
        return bool(is_sky), (votes, votes_thr, a0 % 360.0, a1 % 360.0, min_r)


    def _seg_sky_ratio(self, q, seg_map, H_img: int, W_img: int, sky_labels):
        """
        Проверка "небо" по сегментации: доля пикселей внутри quad, равных sky_labels.
        """
        import numpy as np
        import cv2

        if seg_map is None or (not hasattr(seg_map, "shape")):
            return None
        if seg_map.shape[0] != H_img or seg_map.shape[1] != W_img:
            return None

        q = np.asarray(q, dtype=np.float32).reshape(4, 2)
        mask = np.zeros((H_img, W_img), dtype=np.uint8)
        qi = np.round(q).astype(np.int32)
        qi[:, 0] = np.clip(qi[:, 0], 0, W_img - 1)
        qi[:, 1] = np.clip(qi[:, 1], 0, H_img - 1)
        cv2.fillConvexPoly(mask, qi, 1)

        idx = (mask == 1)
        denom = int(idx.sum())
        if denom <= 0:
            return 0.0

        seg_vals = seg_map[idx]

        if isinstance(sky_labels, (set, list, tuple)):
            sky_labels = np.array(list(sky_labels), dtype=seg_vals.dtype)
        elif isinstance(sky_labels, np.ndarray):
            pass
        else:
            sky_labels = np.array([int(sky_labels)], dtype=seg_vals.dtype)

        sky = np.isin(seg_vals, sky_labels)
        return float(sky.mean())


    def _sky_ban(self, q, H_img: int, W_img: int, debug: bool) -> bool:
        """
        Единая точка принятия решения "пропускать ли этот quad как небо".
        """
        sky_like, met = self._is_sky_like_quad_geom(q, H_img, W_img)
        if sky_like:
            if debug:
                area, yc, bw, bh = met
                print(f"[OVERLAY] SKIP SKY(geom): area={area:.3f} yc={yc:.3f} bw={bw:.3f} bh={bh:.3f}")
            return True

        use_seg_check = bool(getattr(self, "overlay_use_seg_sky_check", True))
        if use_seg_check:
            seg_map = getattr(self, "_cur_seg", None)
            if seg_map is None:
                seg_map = getattr(self, "cur_seg", None)
            if seg_map is None:
                seg_map = getattr(self, "seg", None)

            sky_labels = getattr(self, "overlay_sky_labels", {1})
            ratio_thr = float(getattr(self, "overlay_sky_label_ratio_thr", 0.60))
            r = self._seg_sky_ratio(q, seg_map, H_img, W_img, sky_labels)
            if r is not None and r >= ratio_thr:
                if debug:
                    print(f"[OVERLAY] SKIP SKY(seg): sky_ratio={r:.3f} thr={ratio_thr:.3f} labels={sky_labels}")
                return True

        is_sky_sec, sec_info = self._sky_sector_vote(q, H_img, W_img)
        if is_sky_sec:
            if debug and sec_info is not None:
                votes, thr, a0, a1, min_r = sec_info
                print(f"[OVERLAY] SKIP SKY(sector): votes={votes}/{thr} sector=[{a0:.1f}..{a1:.1f}] min_r={min_r:.2f}")
            return True

        return False
    def _overlay_order_quad_tl_tr_br_bl(self, quad):
        """
        Приводит quad к порядку: TL, TR, BR, BL (image-space, y вниз).
        Надёжный вариант через sum/diff.
        """
        import numpy as np
        q = np.asarray(quad, dtype=np.float32).reshape(4, 2)

        s = q[:, 0] + q[:, 1]          # x+y
        d = q[:, 0] - q[:, 1]          # x-y  (ВАЖНО!)

        tl = q[np.argmin(s)]
        br = q[np.argmax(s)]
        tr = q[np.argmax(d)]           # <-- было перепутано
        bl = q[np.argmin(d)]           # <-- было перепутано

        return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)



    def _norm180(self, a):
        a = float(a)
        return (a + 180.0) % 360.0 - 180.0


    def _clamp_readable_angle(self, a):
        """
        Приводит угол к диапазону [-90, 90] (эквивалентные направления считаем одинаковыми).
        """
        a = self._norm180(a)
        if a > 90.0:
            a -= 180.0
        elif a < -90.0:
            a += 180.0
        return float(a)


    def _dbg_hgeom(self, lbl, place_mask, H, Hinv, img_shape_hw, *, tag=""):

        H_img, W_img = img_shape_hw

        if place_mask is None or H is None or Hinv is None:
            print(f"[HGEOM]{tag} label={lbl}: missing place_mask/H/Hinv")
            return

        pm = np.asarray(place_mask)
        if pm.ndim != 2:
            print(f"[HGEOM]{tag} label={lbl}: bad place_mask ndim={pm.ndim}")
            return

        free = (pm == 0)
        free_area = int(free.sum())
        if free_area == 0:
            print(f"[HGEOM]{tag} label={lbl}: free_area=0 (mask has no zeros)")
            return

        ys, xs = np.where(free)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        W_fp = int(pm.shape[1]); H_fp = int(pm.shape[0])

        # базовая санити матриц
        Hm = np.asarray(H, dtype=np.float64)
        Hi = np.asarray(Hinv, dtype=np.float64)
        if Hm.shape != (3,3) or Hi.shape != (3,3) or (not np.isfinite(Hm).all()) or (not np.isfinite(Hi).all()):
            print(f"[HGEOM]{tag} label={lbl}: bad H/Hinv shapes or NaN/Inf")
            return

        # check identity: H * Hinv ~ I
        I = Hm @ Hi
        Ierr = float(np.linalg.norm(I - np.eye(3), ord='fro'))

        # condition numbers (очень полезно понимать “срыв” перспективы)
        try:
            condH = float(np.linalg.cond(Hm))
            condHi = float(np.linalg.cond(Hi))
        except Exception:
            condH, condHi = float("nan"), float("nan")

        # sample points in FP free area -> warp to image via Hinv -> back via H
        npts = int(getattr(self, "debug_hgeom_npts", 64))
        N = int(xs.size)
        pick = min(npts, N)
        idx = np.random.choice(N, pick, replace=False)

        fp_pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)  # (N,2) as (x,y)

        fp_pts_cv = fp_pts.reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(fp_pts_cv, Hi).reshape(-1, 2)  # FP->IMG
        fp_back = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), Hm).reshape(-1, 2)  # IMG->FP

        reproj = np.linalg.norm(fp_back - fp_pts, axis=1)
        reproj_mean = float(np.mean(reproj))
        reproj_p95  = float(np.percentile(reproj, 95))
        reproj_max  = float(np.max(reproj))

        inside = (
            (img_pts[:, 0] >= -0.5) & (img_pts[:, 0] <= W_img - 0.5) &
            (img_pts[:, 1] >= -0.5) & (img_pts[:, 1] <= H_img - 0.5)
        )
        inside_frac = float(np.mean(inside))

        if img_pts.shape[0] > 0:
            ix0, ix1 = float(np.min(img_pts[:, 0])), float(np.max(img_pts[:, 0]))
            iy0, iy1 = float(np.min(img_pts[:, 1])), float(np.max(img_pts[:, 1]))
        else:
            ix0=ix1=iy0=iy1=float("nan")

        print(
            f"[HGEOM]{tag} label={int(lbl)} | FPmask={W_fp}x{H_fp} free_area={free_area} "
            f"fp_bbox=({x0},{y0})-({x1},{y1}) | "
            f"IMGbbox~({ix0:.1f},{iy0:.1f})-({ix1:.1f},{iy1:.1f}) inside={inside_frac*100:.1f}% | "
            f"Ierr={Ierr:.3e} condH={condH:.2e} condHinv={condHi:.2e} | "
            f"reproj(px): mean={reproj_mean:.3f} p95={reproj_p95:.3f} max={reproj_max:.3f}"
        )

        if bool(getattr(self, "debug_hgeom_print_mats", False)):
            print("[HGEOM] H (IMG->FP):\n", Hm)
            print("[HGEOM] Hinv (FP->IMG):\n", Hi)


    def _sample_layout_text(self, nline, nchar, max_retries=20):
        """
        Compatibility wrapper around TextRenderingService.

        The queue, language tracking and corpus sampling live in
        `self.text_service`; this method stays here because RendererV3 already
        calls it as part of the placement flow.
        """
        min_len = int(getattr(self, "min_word_len", 4))
        layout_text = self.text_service.sample_layout_text(
            nline,
            nchar,
            min_word_len=min_len,
            max_retries=max_retries,
        )
        return layout_text.text, layout_text.lang


    def estimate_region_angle_from_seg(self, seg, lbl):
        """
        Оценивает угол длинной оси сегмента В КООРДИНАТАХ ИЗОБРАЖЕНИЯ.
        Надёжнее, чем через FP/Hinv, потому что seg точно в image-space.

        Возвращает угол в диапазоне [-90, 90] (читаемый).
        """

        if seg is None:
            return 0.0

        seg = np.asarray(seg)
        ys, xs = np.where(seg == int(lbl))
        if xs.size < 20:
            return 0.0

        pts = np.c_[xs.astype(np.float32), ys.astype(np.float32)]  # (x,y)
        (cx, cy), (w, h), theta = cv2.minAreaRect(pts)

        # угол вдоль длинной стороны
        if w < h:
            theta += 90.0

        # в читаемый диапазон [-90, 90]
        if theta > 90.0:
            theta -= 180.0
        if theta < -90.0:
            theta += 180.0

        return float(theta)


    # === ВСТАВЬ ВНУТРЬ class RendererV3 (замени существующий render_text_overlay и связанные helper'ы) ===

    def _overlay_resolve_target_quad(self, region_coords):
        """
        Возвращает dst_quad: (4,2) float32 в image-space + порядок TL,TR,BR,BL
        """
        import numpy as np
        rc = np.asarray(region_coords, dtype=np.float32)

        quad = None
        if rc.ndim == 2 and rc.shape == (4, 2):
            quad = rc
        elif rc.ndim == 2 and rc.shape[1] == 2 and rc.shape[0] >= 4:
            xs, ys = rc[:, 0], rc[:, 1]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        else:
            flat = np.asarray(region_coords, dtype=np.float32).ravel()
            if flat.size == 4:
                a, b, c, d = [float(v) for v in flat.tolist()]
                quad = np.array([[a, b], [c, b], [c, d], [a, d]], dtype=np.float32)

        if quad is None:
            return None

        quad = self._overlay_order_quad_tl_tr_br_bl(quad)
        return quad


    def _overlay_canvas_size_from_quad(self, dst_quad, min_size=64, max_size=1800, scale=1.0):
        import numpy as np
        import math

        q = np.array(dst_quad, dtype=np.float32).reshape(4, 2)  # TL,TR,BR,BL

        def _len(a, b):
            v = b - a
            return float(math.hypot(float(v[0]), float(v[1])))

        w_top = _len(q[0], q[1])
        w_bot = _len(q[3], q[2])
        h_lft = _len(q[0], q[3])
        h_rgt = _len(q[1], q[2])

        # ВАЖНО: берем MAX, иначе при сильной перспективе w_top становится маленьким и рендер "умирает"
        w = max(w_top, w_bot)
        h = max(h_lft, h_rgt)

        Wc = int(np.clip(w * float(scale), float(min_size), float(max_size)))
        Hc = int(np.clip(h * float(scale), float(min_size), float(max_size)))
        return Wc, Hc


    def _overlay_fit_rgba_into_canvas(self, rgb, a, fill=0.70, thr=8, allow_upscale=True, max_up=1.25):
        """
        Масштабирует содержимое (текст) внутри локального канваса так,
        чтобы bbox alpha занимал примерно fill долю канваса.
        """

        H, W = a.shape[:2]
        m = (a > thr)
        if not m.any():
            return rgb, a

        ys, xs = np.where(m)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)

        target_w = max(1, int(W * float(fill)))
        target_h = max(1, int(H * float(fill)))

        s = min(target_w / float(bw), target_h / float(bh))
        if not allow_upscale:
            s = min(1.0, s)
        else:
            s = min(float(max_up), s)

        # если масштаб почти 1 — не трогаем
        if 0.97 <= s <= 1.03:
            return rgb, a

        new_w = max(1, int(round(bw * s)))
        new_h = max(1, int(round(bh * s)))

        crop_rgb = rgb[y0:y1+1, x0:x1+1]
        crop_a   = a[y0:y1+1, x0:x1+1]

        crop_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
        crop_a   = cv2.resize(crop_a,   (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        out_rgb = np.zeros((H, W, 3), dtype=rgb.dtype)
        out_a   = np.zeros((H, W), dtype=a.dtype)

        ox = (W - new_w) // 2
        oy = (H - new_h) // 2

        out_rgb[oy:oy+new_h, ox:ox+new_w] = crop_rgb
        out_a[oy:oy+new_h,   ox:ox+new_w] = crop_a
        return out_rgb, out_a


    def _overlay_render_text_pygame_rgba(self, txt_str, font, Wc, Hc, max_shrink_iters=14):
        import numpy as np
        import pygame
        import random

        self._pygame_init_once()

        Wc = int(Wc); Hc = int(Hc)
        surf = self._overlay_get_surface(Wc, Hc)

        try:
            font.origin = True
        except Exception:
            pass

        fill_w_min = float(getattr(self, "overlay_fill_w_min", 0.60))
        fill_w_max = float(getattr(self, "overlay_fill_w_max", 0.92))
        fill_h_min = float(getattr(self, "overlay_fill_h_min", 0.30))
        fill_h_max = float(getattr(self, "overlay_fill_h_max", 0.62))

        grow_factor = float(getattr(self, "overlay_font_grow_factor", 1.18))
        grow_iters  = int(getattr(self, "overlay_font_grow_iters", 28))
        grow_max_ratio = float(getattr(self, "overlay_font_grow_ratio_max", 8.0))

        grow_factor = max(1.02, min(1.30, grow_factor))
        grow_iters = max(0, min(80, grow_iters))
        grow_max_ratio = max(1.0, min(20.0, grow_max_ratio))

        min_start = int(getattr(self, "overlay_min_font_start_px", 0))

        def get_rect_safe():
            try:
                return font.get_rect(txt_str)
            except Exception:
                return None

        fill_w_min = max(0.20, min(0.95, fill_w_min))
        fill_w_max = max(fill_w_min, min(0.98, fill_w_max))
        fill_h_min = max(0.10, min(0.90, fill_h_min))
        fill_h_max = max(fill_h_min, min(0.95, fill_h_max))

        target_w_min = Wc * float(fill_w_min)
        target_h_min = Hc * float(fill_h_min)
        target_w_max = Wc * float(fill_w_max)
        target_h_max = Hc * float(fill_h_max)

        start_size = int(getattr(font, "size", 24))
        start_size = max(4, start_size)
        if min_start > 0:
            start_size = max(start_size, int(min_start))
        max_size = max(4, int(round(start_size * grow_max_ratio)))

        font.size = start_size

        # GROW (AND условие)
        for _ in range(grow_iters):
            text_rect = get_rect_safe()
            if text_rect is None:
                return None, None, 0
            if (text_rect.width >= target_w_min) and (text_rect.height >= target_h_min):
                break
            cur = int(getattr(font, "size", start_size))
            nxt = int(round(cur * grow_factor))
            if nxt <= cur:
                nxt = cur + 1
            if nxt > max_size:
                break
            font.size = nxt

        text_rect = get_rect_safe()
        if text_rect is None:
            return None, None, 0
        if (text_rect.width < target_w_min) or (text_rect.height < target_h_min):
            return None, None, 0

        # SHRINK
        for _ in range(int(max_shrink_iters)):
            text_rect = get_rect_safe()
            if text_rect is None:
                return None, None, 0
            if (text_rect.width <= target_w_max) and (text_rect.height <= target_h_max):
                break
            cur = int(getattr(font, "size", start_size))
            nxt = max(4, int(cur * 0.90))
            if nxt >= cur:
                nxt = cur - 1
            if nxt < 4:
                return None, None, 0
            font.size = nxt

        text_rect = get_rect_safe()
        if text_rect is None:
            return None, None, 0

        tx = (Wc - text_rect.width) // 2
        ty = (Hc - text_rect.height) // 2 + text_rect.height

        bright = [
            (255, 255, 255), (255, 255, 0), (0, 255, 255),
            (255, 128, 0), (0, 255, 0), (255, 0, 255), (0, 128, 255),
        ]
        fg = random.choice(bright)

        try:
            font.render_to(surf, (int(tx), int(ty)), txt_str, fg)
        except Exception:
            return None, None, 0

        rgb = pygame.surfarray.pixels3d(surf).copy().swapaxes(0, 1)
        a   = pygame.surfarray.pixels_alpha(surf).copy().swapaxes(0, 1)

        if int(a.sum()) < 10:
            return None, None, 0

        n_chars = sum(1 for c in txt_str if not c.isspace())
        return rgb.astype(np.uint8), a.astype(np.uint8), int(n_chars)


    def _overlay_rotate_rgba(self, rgb, a, angle_deg):
        """Поворачиваем локальный RGBA вокруг центра."""
        if rgb is None or a is None:
            return rgb, a
        try:
            ang = float(angle_deg)
        except Exception:
            ang = 0.0
        if abs(ang) <= 0.5:
            return rgb, a

        h, w = a.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        rgb_r = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        a_r = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rgb_r, a_r


    def _overlay_add_outline_local(self, rgb, a, outline_iters=2):
        """Добавляем чёрную обводку в локальном RGBA перед варпом."""
        if rgb is None or a is None:
            return rgb, a
        m = (a > 0).astype(np.uint8) * 255
        if int(m.sum()) == 0:
            return rgb, a

        k = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(m, k, iterations=int(outline_iters))
        outline = (dil > 0) & (m == 0)

        rgb2 = rgb.copy()
        a2 = a.copy()
        rgb2[outline] = (0, 0, 0)
        a2[outline] = 255
        return rgb2, a2

    def _overlay_warp_rgba_to_image(self, rgb, a, dst_quad, out_shape_hw):
        """
        Варп RGBA из src-rect в dst_quad (image-space).

        FIX для сильной перспективы:
        - RGB варпим INTER_LINEAR (нормально)
        - ALPHA варпим INTER_NEAREST (иначе альфа при сильном сжатии "размазывается" и исчезает)
        """

        H_img, W_img = out_shape_hw
        h, w = a.shape[:2]

        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst = dst_quad.astype(np.float32)

        M = cv2.getPerspectiveTransform(src, dst)

        warped_rgb = cv2.warpPerspective(
            rgb, M, (W_img, H_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        # ключевое: nearest для альфы
        warped_a = cv2.warpPerspective(
            a, M, (W_img, H_img),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # опционально: чуть "поджирнить" альфу после варпа (если хочется)
        dil = int(getattr(self, "overlay_alpha_dilate", 0))
        if dil > 0:
            k = np.ones((3, 3), np.uint8)
            warped_a = cv2.dilate(warped_a, k, iterations=dil)

        return warped_rgb, warped_a



    def _overlay_apply_bg_rect_imgspace(self, img, warped_a, pad_px=16, alpha_thr=10):
        """
        Фон под текстом: bbox по альфе (image-space), цвет = mean под bbox.

        FIX:
        - alpha_thr теперь параметр (по умолчанию 10), чтобы согласовывать с overlay_alpha_thr.
        """

        if img is None or warped_a is None:
            return img

        thr = int(alpha_thr)
        thr = max(1, thr)

        m = (warped_a > thr).astype(np.uint8) * 255
        ys, xs = np.where(m > 0)
        if xs.size < 10:
            return img

        x, y, w, h = cv2.boundingRect(np.stack([xs, ys], axis=1).astype(np.int32))
        x0 = max(0, x - int(pad_px))
        y0 = max(0, y - int(pad_px))
        x1 = min(img.shape[1], x + w + int(pad_px))
        y1 = min(img.shape[0], y + h + int(pad_px))

        if (x1 - x0) < 2 or (y1 - y0) < 2:
            return img

        roi = img[y0:y1, x0:x1]
        if roi.ndim == 3 and roi.shape[2] == 3:
            mean_rgb = roi.reshape(-1, 3).astype(np.float32).mean(axis=0)
            bg = tuple(int(np.clip(np.round(v), 0, 255)) for v in mean_rgb)
            out = img.copy()
            out[y0:y1, x0:x1] = bg
            return out

        return img



    def _overlay_alpha_blend(self, base_img, over_rgb, over_a):
        """Альфа-бленд в RGB (base_img assumed RGB)."""
        import numpy as np
        a = (over_a.astype(np.float32) / 255.0)
        if a.max() <= 0.0:
            return base_img
        a3 = a[:, :, None]
        out = base_img.astype(np.float32) * (1.0 - a3) + over_rgb.astype(np.float32) * a3
        return np.clip(out, 0, 255).astype(base_img.dtype)


    def _overlay_build_charBB_from_mask(self, text_mask_img, n_chars):
        """PCA-charBB по глобальной маске, возвращает (2,4,N) или None."""
        import numpy as np

        n_chars = int(n_chars or 0)
        if n_chars <= 0:
            return None

        ys, xs = np.where(text_mask_img > 0)
        if xs.size == 0 or ys.size == 0:
            return None

        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        mean = pts.mean(axis=0)
        pc = pts - mean[None, :]

        cov = np.cov(pc, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        d0 = evecs[:, int(np.argmax(evals))].astype(np.float32)
        n = float(np.linalg.norm(d0))
        if n < 1e-6:
            d0 = np.array([1.0, 0.0], dtype=np.float32)
        else:
            d0 /= n
        if d0[0] < 0:
            d0 = -d0
        d1 = np.array([-d0[1], d0[0]], dtype=np.float32)

        us = pc @ d0
        vs = pc @ d1
        umin, umax = float(us.min()), float(us.max())
        vmin, vmax = float(vs.min()), float(vs.max())

        du = max(1e-6, umax - umin)
        dv = max(1e-6, vmax - vmin)
        umin -= 0.02 * du; umax += 0.02 * du
        vmin -= 0.05 * dv; vmax += 0.05 * dv

        charBB = np.zeros((2, 4, n_chars), dtype=np.float32)
        for i in range(n_chars):
            t0 = float(i) / n_chars
            t1 = float(i + 1) / n_chars
            u0 = umin + t0 * (umax - umin)
            u1 = umin + t1 * (umax - umin)

            p0 = mean + u0 * d0 + vmin * d1
            p1 = mean + u1 * d0 + vmin * d1
            p2 = mean + u1 * d0 + vmax * d1
            p3 = mean + u0 * d0 + vmax * d1

            quad = np.stack([p0, p1, p2, p3], axis=0)
            charBB[0, :, i] = quad[:, 0]
            charBB[1, :, i] = quad[:, 1]

        H, W = text_mask_img.shape[:2]
        charBB[0] = np.clip(charBB[0], 0, W - 1)
        charBB[1] = np.clip(charBB[1], 0, H - 1)
        return charBB

    def _mask_bbox(self, m: np.ndarray):
        """bbox по маске (uint8/bool). Возвращает (x0,y0,x1,y1) inclusive, либо None."""
        import numpy as np
        if m is None or m.size == 0:
            return None
        if m.dtype != np.bool_:
            idx = m > 0
        else:
            idx = m
        ys, xs = np.where(idx)
        if len(xs) == 0:
            return None
        x0 = int(xs.min()); x1 = int(xs.max())
        y0 = int(ys.min()); y1 = int(ys.max())
        return x0, y0, x1, y1


    def _shift_image_reflect(self, img: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Сдвиг картинки, края отражаем."""
        import cv2
        import numpy as np
        H, W = img.shape[:2]
        M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


    def _apply_gain_gamma_rgb(self, img: np.ndarray, gain: float, gamma: float) -> np.ndarray:
        """Лёгкая тональная правка окклюдера."""
        import numpy as np
        x = img.astype(np.float32) / 255.0
        x = np.clip(x * float(gain), 0.0, 1.0)
        x = np.power(x, float(gamma))
        return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


    def _occ_make_shape_mask(self, H: int, W: int, bbox, kind: str) -> np.ndarray:
        """
        Делает одну фигуру-окклюдер в full-res маске.
        kind: 'band' | 'ellipse' | 'poly'
        """
        import numpy as np
        import cv2
        import math

        x0, y0, x1, y1 = bbox
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)

        m = np.zeros((H, W), dtype=np.uint8)

        # центр около bbox, чтобы окклюдер реально пересекал текст
        cx = int(x0 + 0.5 * bw + np.random.randint(-bw // 6, bw // 6 + 1))
        cy = int(y0 + 0.5 * bh + np.random.randint(-bh // 6, bh // 6 + 1))
        cx = int(np.clip(cx, 0, W - 1))
        cy = int(np.clip(cy, 0, H - 1))

        diag = math.hypot(bw, bh) + 1e-6

        if kind == "band":
            # длинная полоска (типично: провод/ветка/ремень)
            angle = float(np.random.uniform(0, 180))
            length = float(np.random.uniform(0.9, 1.4)) * diag
            width = float(np.random.uniform(0.06, 0.16)) * min(bw, bh)
            width = max(2.0, width)

            rect = ((float(cx), float(cy)), (float(length), float(width)), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            box[:, 0] = np.clip(box[:, 0], 0, W - 1)
            box[:, 1] = np.clip(box[:, 1], 0, H - 1)
            cv2.fillConvexPoly(m, box, 255)

        elif kind == "ellipse":
            # пятно (типично: листья/грязь/капля/блик)
            ax1 = int(max(2, np.random.uniform(0.10, 0.30) * bw))
            ax2 = int(max(2, np.random.uniform(0.10, 0.30) * bh))
            angle = float(np.random.uniform(0, 180))
            cv2.ellipse(m, (cx, cy), (ax1, ax2), angle, 0, 360, 255, thickness=-1)

        else:  # "poly"
            # неровный многоугольник
            n = int(np.random.randint(4, 7))
            pts = []
            for _ in range(n):
                px = int(cx + np.random.uniform(-0.35, 0.35) * bw)
                py = int(cy + np.random.uniform(-0.35, 0.35) * bh)
                pts.append([int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))])
            pts = np.array(pts, dtype=np.int32)
            cv2.fillConvexPoly(m, pts, 255)

        return m


    def _overlay_apply_synth_occlusion(self, base_img, warped_rgb, warped_a_full, alpha_thr=2):
        import numpy as np
        import cv2
        import math

        H, W = warped_a_full.shape[:2]
        log = bool(getattr(self, "overlay_occ_log", False))

        alpha_thr = int(alpha_thr)
        text_mask = (warped_a_full > alpha_thr)
        text_area = int(text_mask.sum())

        if log:
            print(f"[OCC] start: HxW={H}x{W} alpha_thr={alpha_thr} text_area={text_area}")

        if text_area <= 0:
            if log:
                print("[OCC] skip: empty text mask")
            return warped_rgb, warped_a_full, None, None

        # Сделаем срабатывание чаще (ты можешь переопределять снаружи)
        p_occ = float(getattr(self, "overlay_occ_p", 0.65))
        r = float(np.random.rand())
        if log:
            print(f"[OCC] gate: p_occ={p_occ:.3f} rand={r:.3f}")
        if r > p_occ:
            if log:
                print("[OCC] gate: not triggered")
            return warped_rgb, warped_a_full, None, None

        ys, xs = np.where(text_mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        if bw < 6 or bh < 6:
            if log:
                print(f"[OCC] skip: tiny bbox bw={bw} bh={bh}")
            return warped_rgb, warped_a_full, None, None

        # expanded area (чтобы окклюдер мог чуть выходить за текст)
        expand_px = int(getattr(self, "overlay_occ_expand_px", 6))
        expand_px = max(0, min(64, expand_px))
        expanded = (text_mask.astype(np.uint8) * 255)
        if expand_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1))
            expanded = cv2.dilate(expanded, k, iterations=1)
        expanded_bool = (expanded > 0)
        if log:
            print(f"[OCC] expand: expand_px={expand_px} expand_area={int(expanded_bool.sum())}")

        # Суммарная цель перекрытия
        frac_min = float(getattr(self, "overlay_occ_frac_min", 0.08))
        frac_max = float(getattr(self, "overlay_occ_frac_max", 0.18))
        frac_min = max(0.0, min(0.9, frac_min))
        frac_max = max(frac_min, min(0.95, frac_max))
        target_frac = float(np.random.uniform(frac_min, frac_max))
        target_cover = int(round(target_frac * text_area))

        max_cover_frac = float(getattr(self, "overlay_occ_max_cover_frac", 0.35))
        max_cover_frac = max(0.05, min(0.95, max_cover_frac))
        max_cover = int(round(max_cover_frac * text_area))

        if log:
            print(f"[OCC] target: frac~U({frac_min:.3f},{frac_max:.3f}) -> target_frac={target_frac:.3f}")
            print(f"[OCC] target_cover={target_cover}  max_cover={max_cover} (max_cover_frac={max_cover_frac:.3f})")

        # Сколько кусочков окклюзии на один текст
        pieces_min, pieces_max = getattr(self, "overlay_occ_pieces_range", (2, 4))
        try:
            pieces_min = int(pieces_min)
            pieces_max = int(pieces_max)
        except Exception:
            pieces_min, pieces_max = 2, 4
        pieces_min = max(1, min(8, pieces_min))
        pieces_max = max(pieces_min, min(10, pieces_max))

        # Если текст маленький — не делаем слишком много кусочков
        if text_area < 4000:
            pieces_max = min(pieces_max, 3)
        if text_area < 2000:
            pieces_max = min(pieces_max, 2)

        n_pieces = int(np.random.randint(pieces_min, pieces_max + 1))

        # Размер одного кусочка (как доля текста)
        piece_frac_min = float(getattr(self, "overlay_occ_piece_frac_min", 0.015))
        piece_frac_max = float(getattr(self, "overlay_occ_piece_frac_max", 0.055))
        piece_frac_min = max(0.001, min(0.25, piece_frac_min))
        piece_frac_max = max(piece_frac_min, min(0.35, piece_frac_max))

        # Минимум пикселей на кусочек, чтобы не было "пыли"
        min_piece_px = int(getattr(self, "overlay_occ_min_piece_px", 140))
        min_piece_px = max(20, min(3000, min_piece_px))

        # Общие параметры форм
        probs = getattr(self, "overlay_occ_kind_probs", None)
        if not isinstance(probs, dict):
            # меньше "полос", больше "стикеров/эллипсов"
            probs = {"band_poly": 0.18, "sticker": 0.42, "ellipse": 0.30, "edge_block": 0.10}

        kinds = [k for k in probs.keys()]
        weights = np.array([float(probs[k]) for k in kinds], dtype=np.float32)
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 1e-6:
            weights[:] = 1.0
        weights /= weights.sum()

        diag = float(math.hypot(bw, bh))
        min_dim = float(min(bw, bh))

        # Допуски/попытки
        tries_per_piece = int(getattr(self, "overlay_occ_tries_per_piece", 10))
        tries_per_piece = max(2, min(40, tries_per_piece))

        close_thr = getattr(self, "overlay_occ_close_thr_px", 350)
        try:
            close_thr = int(close_thr)
        except Exception:
            try:
                close_thr = int(close_thr[0])
            except Exception:
                close_thr = 350
        close_thr = max(50, min(5000, close_thr))

        # ВАЖНО: не даём кусочкам сливаться в одно большое
        avoid_overlap = bool(getattr(self, "overlay_occ_avoid_overlap", True))

        occ_total = np.zeros((H, W), np.uint8)
        covered_total = 0

        # подготовим список индексов текстовых пикселей для центров
        idx = np.arange(len(xs), dtype=np.int32)

        if log:
            print(f"[OCC] pieces: n_pieces={n_pieces} piece_frac~U({piece_frac_min:.3f},{piece_frac_max:.3f}) "
                f"min_piece_px={min_piece_px} tries_per_piece={tries_per_piece}")

        for pi in range(n_pieces):
            # целевой размер кусочка
            # сначала берём долю, потом ограничиваем сверху, чтобы суммарно не улететь
            piece_frac = float(np.random.uniform(piece_frac_min, piece_frac_max))
            piece_target = int(round(piece_frac * text_area))
            piece_target = max(min_piece_px, piece_target)

            # если осталось мало до target_cover — подожмём
            remaining = max(0, target_cover - covered_total)
            if remaining > 0:
                piece_target = min(piece_target, max(min_piece_px, remaining))
            else:
                # уже перекрыли цель — можно остановиться
                break

            best_piece = None
            best_kind = None
            best_new_cover = 0
            best_diff = 10**18

            # выбираем центр по случайному пикселю текста
            c_i = int(np.random.choice(idx))
            cx0 = int(xs[c_i])
            cy0 = int(ys[c_i])

            for t in range(tries_per_piece):
                kind = str(np.random.choice(kinds, p=weights))
                occ = np.zeros((H, W), np.uint8)

                # случайный центр рядом с выбранным
                jx = int(np.random.randint(-max(2, bw // 10), max(3, bw // 10)))
                jy = int(np.random.randint(-max(2, bh // 10), max(3, bh // 10)))
                cx = int(np.clip(cx0 + jx, x0, x1))
                cy = int(np.clip(cy0 + jy, y0, y1))

                if kind == "band_poly":
                    # короткая/тонкая "полоска" (не вечная одна и та же)
                    base_ang = float(np.random.uniform(-70.0, 70.0))
                    ang = math.radians(base_ang)

                    # длина меньше, чем раньше
                    L = diag * float(np.random.uniform(0.45, 0.95))
                    dx = math.cos(ang) * L * 0.5
                    dy = math.sin(ang) * L * 0.5

                    p0 = np.array([cx - dx, cy - dy], dtype=np.float32)
                    p2 = np.array([cx + dx, cy + dy], dtype=np.float32)

                    perp = np.array([-math.sin(ang), math.cos(ang)], dtype=np.float32)
                    bend = float(np.random.uniform(-0.18, 0.18)) * min_dim
                    pm = (p0 + p2) * 0.5 + perp * bend

                    pts = np.stack([p0, pm, p2], axis=0)
                    pts_i = np.round(pts).astype(np.int32)

                    thick = float(np.random.uniform(0.04, 0.12)) * min_dim
                    thick_px = int(max(2, round(thick)))

                    cv2.polylines(occ, [pts_i], isClosed=False, color=255, thickness=thick_px, lineType=cv2.LINE_AA)

                elif kind == "sticker":
                    rw = float(np.random.uniform(0.16, 0.38)) * bw
                    rh = float(np.random.uniform(0.10, 0.26)) * bh
                    angle = float(np.random.uniform(-60.0, 60.0))

                    rect = ((float(cx), float(cy)), (max(6.0, rw), max(6.0, rh)), angle)
                    box = cv2.boxPoints(rect).astype(np.int32)
                    cv2.fillConvexPoly(occ, box, 255, lineType=cv2.LINE_AA)

                    blur = int(getattr(self, "overlay_occ_soft_blur", 3))
                    blur = max(0, min(9, blur))
                    if blur >= 3 and (blur % 2 == 1):
                        tmp = cv2.GaussianBlur(occ, (blur, blur), 0)
                        occ = (tmp > 40).astype(np.uint8) * 255

                elif kind == "ellipse":
                    ax = int(max(3, round(np.random.uniform(0.10, 0.26) * bw)))
                    ay = int(max(3, round(np.random.uniform(0.08, 0.22) * bh)))
                    angle = float(np.random.uniform(-70.0, 70.0))
                    cv2.ellipse(occ, (cx, cy), (ax, ay), angle, 0.0, 360.0, 255, thickness=-1, lineType=cv2.LINE_AA)

                else:  # edge_block — маленький "срез" у края bbox текста
                    side = int(np.random.choice([0, 1, 2, 3]))
                    if side == 0:  # left
                        xx0 = max(0, x0 - int(0.10 * bw))
                        xx1 = int(round(x0 + np.random.uniform(0.10, 0.22) * bw))
                        yy0 = int(round(cy - np.random.uniform(0.20, 0.35) * bh))
                        yy1 = int(round(cy + np.random.uniform(0.20, 0.35) * bh))
                    elif side == 1:  # right
                        xx0 = int(round(x1 - np.random.uniform(0.22, 0.10) * bw))  # small chunk
                        xx1 = min(W - 1, x1 + int(0.10 * bw))
                        yy0 = int(round(cy - np.random.uniform(0.20, 0.35) * bh))
                        yy1 = int(round(cy + np.random.uniform(0.20, 0.35) * bh))
                    elif side == 2:  # top
                        yy0 = max(0, y0 - int(0.10 * bh))
                        yy1 = int(round(y0 + np.random.uniform(0.10, 0.22) * bh))
                        xx0 = int(round(cx - np.random.uniform(0.20, 0.35) * bw))
                        xx1 = int(round(cx + np.random.uniform(0.20, 0.35) * bw))
                    else:  # bottom
                        yy0 = int(round(y1 - np.random.uniform(0.22, 0.10) * bh))
                        yy1 = min(H - 1, y1 + int(0.10 * bh))
                        xx0 = int(round(cx - np.random.uniform(0.20, 0.35) * bw))
                        xx1 = int(round(cx + np.random.uniform(0.20, 0.35) * bw))

                    xx0 = int(np.clip(xx0, 0, W - 1))
                    xx1 = int(np.clip(xx1, 0, W - 1))
                    yy0 = int(np.clip(yy0, 0, H - 1))
                    yy1 = int(np.clip(yy1, 0, H - 1))
                    if xx1 < xx0: xx0, xx1 = xx1, xx0
                    if yy1 < yy0: yy0, yy1 = yy1, yy0

                    occ[yy0:yy1 + 1, xx0:xx1 + 1] = 255
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    occ = cv2.erode(occ, k, iterations=1)

                # ограничиваем областью expanded и считаем "новое" покрытие
                occ = (occ > 0).astype(np.uint8) * 255
                occ[~expanded_bool] = 0

                if avoid_overlap and covered_total > 0:
                    occ[occ_total > 0] = 0

                new_cover = int(((occ > 0) & text_mask).sum())
                if new_cover <= 0:
                    continue

                # хотим попасть ближе к piece_target (но допускаем небольшую погрешность)
                diff = abs(new_cover - piece_target)

                # ограничение чтобы не выстрелило большим куском
                if new_cover > max_cover:
                    continue

                if diff < best_diff:
                    best_diff = diff
                    best_piece = occ.copy()
                    best_kind = kind
                    best_new_cover = new_cover

                if best_piece is not None and best_diff <= close_thr:
                    break

            if best_piece is None:
                if log:
                    print(f"[OCC] piece {pi+1}/{n_pieces}: failed to make piece")
                continue

            occ_total = cv2.bitwise_or(occ_total, best_piece)
            covered_total += best_new_cover

            if log:
                print(f"[OCC] piece {pi+1}/{n_pieces}: kind={best_kind} cover={best_new_cover} "
                    f"diff={best_diff} covered_total={covered_total}/{target_cover}")

            # если уже достигли цели — можно остановиться
            if covered_total >= target_cover:
                break

        if int((occ_total > 0).sum()) == 0:
            if log:
                print("[OCC] skip: total occ empty")
            return warped_rgb, warped_a_full, None, None

        # Убийство альфы по маске (с небольшим pad)
        kill_pad = int(getattr(self, "overlay_occ_kill_pad_px", 1))
        kill_pad = max(0, min(6, kill_pad))

        kill_mask = occ_total
        if kill_pad > 0:
            kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kill_pad + 1, 2 * kill_pad + 1))
            kill_mask = cv2.dilate(occ_total, kk, iterations=1)

        warped_a_vis = warped_a_full.copy()
        before_nz = int((warped_a_vis > alpha_thr).sum())
        warped_a_vis[kill_mask > 0] = 0
        after_nz = int((warped_a_vis > alpha_thr).sum())

        if log:
            print(f"[OCC] chosen: pieces_total_area={int((occ_total>0).sum())} cover_in_text_total={covered_total} "
                f"alpha_kill: before={before_nz} after={after_nz} killed={before_nz-after_nz}")

        return warped_rgb, warped_a_vis, None, occ_total


    def _occ_parse_ksize(self, v, default=0):
        """
        Принимает:
        - 0/None -> (0,0) (не блюрить)
        - int k -> (k,k)
        - tuple/list (kx,ky) -> (kx,ky)
        Гарантирует odd и >=1 (если не ноль).
        """
        if v is None:
            v = default

        # already tuple/list
        if isinstance(v, (tuple, list)) and len(v) == 2:
            kx, ky = v
            try:
                kx = int(kx)
                ky = int(ky)
            except Exception:
                kx, ky = int(default), int(default)
        else:
            # scalar
            try:
                kx = ky = int(v)
            except Exception:
                kx = ky = int(default)

        if kx <= 0 or ky <= 0:
            return (0, 0)

        # make odd
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1

        # clamp a bit
        kx = max(1, min(51, kx))
        ky = max(1, min(51, ky))
        return (kx, ky)
    
    def _occ_bbox_from_mask(self, m_bool):
        """bbox (x0,y0,x1,y1) inclusive-exclusive; None если пусто"""
        ys, xs = np.where(m_bool)
        if xs.size == 0:
            return None
        x0 = int(xs.min()); x1 = int(xs.max()) + 1
        y0 = int(ys.min()); y1 = int(ys.max()) + 1
        return (x0, y0, x1, y1)


    def render_text_overlay(self, img, txt_str, font, selected_angle, region_coords, depth=None):
        import numpy as np
        import math
        import cv2

        try:
            H_img, W_img = img.shape[:2]
            debug = bool(getattr(self, "debug_hgeom", False))

            # ✅ master-switch: если True — НЕ делаем синтетическую окклюзию (перекрытия)
            disable_all_augs = bool(getattr(self, "disable_all_augs", False))

            dst_quad = self._overlay_resolve_target_quad(region_coords)
            if dst_quad is None:
                if debug:
                    print("[OVERLAY] dst_quad resolve failed")
                return None, None, None

            dst_quad = self._overlay_order_quad_tl_tr_br_bl(dst_quad)
            dst_quad = np.array(dst_quad, dtype=np.float32).reshape(4, 2)

            # SKY BAN
            disallow_sky = bool(getattr(self, "overlay_disallow_sky", True))
            if disallow_sky and self._sky_ban(dst_quad, H_img, W_img, debug):
                return None, None, None

            # tiny quad reject
            min_side_thr = float(getattr(self, "overlay_min_quad_side_px", 22.0))
            min_area_thr = float(getattr(self, "overlay_min_quad_area_px2", 700.0))

            w_top_ = self._edge_len(dst_quad[0], dst_quad[1])
            w_bot_ = self._edge_len(dst_quad[3], dst_quad[2])
            h_lft_ = self._edge_len(dst_quad[0], dst_quad[3])
            h_rgt_ = self._edge_len(dst_quad[1], dst_quad[2])

            w_eff = max(w_top_, w_bot_)
            h_eff = max(h_lft_, h_rgt_)
            area_eff = float(self._poly_area(dst_quad))

            if (min(w_eff, h_eff) < min_side_thr) or (area_eff < min_area_thr):
                if debug:
                    print(
                        f"[OVERLAY] SKIP tiny quad: w_eff={w_eff:.1f} h_eff={h_eff:.1f} area={area_eff:.1f} "
                        f"thr_side={min_side_thr:.1f} thr_area={min_area_thr:.1f}"
                    )
                return None, None, None

            # quad angle
            v = dst_quad[1] - dst_quad[0]
            quad_angle_raw = math.degrees(math.atan2(float(v[1]), float(v[0])))
            quad_angle = self._wrap_deg_pm180(quad_angle_raw)

            try:
                sel = float(selected_angle) if selected_angle is not None else quad_angle
            except Exception:
                sel = quad_angle
            sel = self._wrap_deg_pm180(sel)
            delta = self._wrap_deg_pm180(sel - quad_angle)

            # perspective boost
            boost_param = float(getattr(self, "overlay_persp_boost", 0.0))
            min_ratio = float(getattr(self, "overlay_persp_min_ratio", 0.55))

            if boost_param > 0.0:
                if boost_param < 1.0:
                    boost_factor = 1.0 + 6.0 * boost_param
                else:
                    boost_factor = boost_param
                boost_factor = max(1.0, min(12.0, float(boost_factor)))

                min_ratio_eff = float(min_ratio) ** float(boost_factor)
                min_ratio_eff = max(0.12, min(0.98, float(min_ratio_eff)))

                prefer_axis = str(getattr(self, "overlay_persp_axis", "w")).lower()
                far_by_y = bool(getattr(self, "overlay_persp_far_by_y", True))
                expand_near = bool(getattr(self, "overlay_persp_expand_near", True))

                dst_quad2 = self._apply_persp_boost(
                    dst_quad,
                    boost_factor=boost_factor,
                    min_ratio_eff=min_ratio_eff,
                    prefer_axis=prefer_axis,
                    far_by_y=far_by_y,
                    expand_near=expand_near,
                )

                if np.isfinite(dst_quad2).all() and self._poly_area(dst_quad2) > 25.0:
                    dst_quad = dst_quad2

                dst_quad[:, 0] = np.clip(dst_quad[:, 0], 0.0, float(W_img - 1))
                dst_quad[:, 1] = np.clip(dst_quad[:, 1], 0.0, float(H_img - 1))

            # SKY recheck after boost
            if disallow_sky and self._sky_ban(dst_quad, H_img, W_img, debug):
                return None, None, None

            # require_perspective
            require_perspective = bool(getattr(self, "overlay_require_perspective", False))
            if require_perspective:
                thr = float(getattr(self, "overlay_min_persp_strength", 0.06))
                strength, _, _ = self._persp_strength_from_quad(dst_quad)
                if strength < thr:
                    if debug:
                        print(f"[OVERLAY] SKIP (require_perspective): strength={strength:.3f} < thr={thr:.3f}")
                    return None, None, None

            # local canvas
            min_canvas = int(getattr(self, "overlay_min_canvas_size", 72))
            max_canvas = int(getattr(self, "overlay_max_canvas_size", 1800))
            canvas_scale = float(getattr(self, "overlay_canvas_scale", 1.35))
            canvas_scale = max(1.0, min(2.5, canvas_scale))

            Wc, Hc = self._overlay_canvas_size_from_quad(
                dst_quad, min_size=min_canvas, max_size=max_canvas, scale=canvas_scale
            )

            # render local
            rgb_loc = a_loc = None
            n_chars = 0
            for k in (1.00, 1.25, 1.55):
                Wt = int(min(max_canvas, max(min_canvas, round(Wc * k))))
                Ht = int(min(max_canvas, max(min_canvas, round(Hc * k))))
                rgb_loc, a_loc, n_chars = self._overlay_render_text_pygame_rgba(txt_str, font, Wt, Ht)
                if rgb_loc is not None and a_loc is not None:
                    break

            if rgb_loc is None or a_loc is None:
                if debug:
                    print("[OVERLAY] local render failed/empty")
                return None, None, None

            # rotate only by delta
            rgb_loc, a_loc = self._overlay_rotate_rgba(rgb_loc, a_loc, delta)

            # outline
            outline_iters = int(getattr(self, "overlay_outline_iters", 2))
            strength, _, _ = self._persp_strength_from_quad(dst_quad)
            if strength > 0.8:
                outline_iters = max(outline_iters, 3)
            rgb_loc, a_loc = self._overlay_add_outline_local(rgb_loc, a_loc, outline_iters=outline_iters)

            # fit into canvas
            fill = float(getattr(self, "overlay_text_fill", 0.68))
            max_up = float(getattr(self, "overlay_text_max_up", 1.20))
            rgb_loc, a_loc = self._overlay_fit_rgba_into_canvas(
                rgb_loc, a_loc, fill=fill, thr=8, allow_upscale=True, max_up=max_up
            )

            # warp text to image
            warped_rgb, warped_a_full = self._overlay_warp_rgba_to_image(rgb_loc, a_loc, dst_quad, (H_img, W_img))
            if warped_rgb is None or warped_a_full is None:
                return None, None, None

            if warped_a_full.ndim == 3:
                warped_a_full = warped_a_full[:, :, 0]
            if warped_a_full.dtype != np.uint8:
                warped_a_full = np.clip(warped_a_full, 0, 255).astype(np.uint8)

            alpha_thr = int(getattr(self, "overlay_alpha_thr", 2))
            alpha_thr = max(1, alpha_thr)

            # FULL mask
            text_mask_full = ((warped_a_full > alpha_thr).astype(np.uint8) * 255)
            if int(text_mask_full.sum()) == 0:
                text_mask_full = ((warped_a_full > 1).astype(np.uint8) * 255)
                if int(text_mask_full.sum()) == 0:
                    if debug:
                        print("[OVERLAY] warped mask empty (full)")
                    return None, None, None

            # --- synthetic controlled occlusion (перекрытие) ---
            warped_a_vis = warped_a_full
            occ_mask_u8 = None

            # ✅ если disable_all_augs=True — окклюзию не применяем
            if not disable_all_augs:
                try:
                    warped_rgb, warped_a_vis, _occ_rgb_unused, occ_mask_u8 = self._overlay_apply_synth_occlusion(
                        img, warped_rgb, warped_a_full, alpha_thr=alpha_thr
                    )
                except Exception as e:
                    if debug:
                        print("[OVERLAY] _overlay_apply_synth_occlusion exception:", repr(e))
                    warped_a_vis = warped_a_full
                    occ_mask_u8 = None

            # какую маску возвращать наружу (видимую или полную)
            return_visible_mask = bool(getattr(self, "overlay_return_visible_mask", False))
            if return_visible_mask:
                text_mask_out = ((warped_a_vis > alpha_thr).astype(np.uint8) * 255)
            else:
                text_mask_out = text_mask_full

            if int(text_mask_out.sum()) == 0:
                text_mask_out = ((warped_a_full > 1).astype(np.uint8) * 255)
                if int(text_mask_out.sum()) == 0:
                    if debug:
                        print("[OVERLAY] warped mask empty (out)")
                    return None, None, None

            # --- background UNDER text ---
            # ✅ перспективная СПЛОШНАЯ ПЛАШКА (bbox текста в локалке -> warp тем же dst_quad)
            use_persp_bg = bool(getattr(self, "overlay_bg_perspective", True))

            bg_thr = int(getattr(self, "overlay_bg_alpha_thr", max(10, alpha_thr * 4)))
            bg_thr = max(1, bg_thr)

            img_bg = img

            if use_persp_bg:
                thr_loc = int(getattr(self, "overlay_bg_loc_thr", 8))
                m_loc = (a_loc > thr_loc)

                if m_loc.any():
                    ys, xs = np.where(m_loc)
                    x0l, x1l = int(xs.min()), int(xs.max())
                    y0l, y1l = int(ys.min()), int(ys.max())

                    pad_px = int(getattr(self, "overlay_bg_pad_px", 18))
                    pad_px = max(0, min(200, pad_px))

                    x0l = max(0, x0l - pad_px)
                    y0l = max(0, y0l - pad_px)
                    x1l = min(a_loc.shape[1] - 1, x1l + pad_px)
                    y1l = min(a_loc.shape[0] - 1, y1l + pad_px)

                    bg_a_loc = np.zeros_like(a_loc, dtype=np.uint8)
                    bg_a_loc[y0l:y1l + 1, x0l:x1l + 1] = 255

                    feather = float(getattr(self, "overlay_bg_feather_sigma", 2.0))
                    feather = max(0.0, min(20.0, feather))
                    if feather > 1e-6:
                        bg_a_loc = cv2.GaussianBlur(bg_a_loc, (0, 0), sigmaX=feather, sigmaY=feather)

                    opacity = float(getattr(self, "overlay_bg_opacity", 1.0))
                    opacity = max(0.0, min(1.0, opacity))
                    if opacity < 0.999:
                        bg_a_loc = np.clip(bg_a_loc.astype(np.float32) * opacity, 0, 255).astype(np.uint8)

                    # цвет плашки = mean под quad в исходной сцене (стабильно)
                    poly = np.round(dst_quad).astype(np.int32)
                    poly[:, 0] = np.clip(poly[:, 0], 0, W_img - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, H_img - 1)
                    poly_mask = np.zeros((H_img, W_img), dtype=np.uint8)
                    cv2.fillConvexPoly(poly_mask, poly, 255)
                    mean = cv2.mean(img, mask=poly_mask)[:3]
                    bg = tuple(int(np.clip(round(v), 0, 255)) for v in mean)

                    bg_rgb_loc = np.zeros_like(rgb_loc, dtype=np.uint8)
                    bg_rgb_loc[:, :] = bg

                    # warp плашки: RGB linear, ALPHA linear (чтобы feather не убивался NEAREST)
                    h, w = bg_a_loc.shape[:2]
                    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
                    dst = dst_quad.astype(np.float32)
                    Mbg = cv2.getPerspectiveTransform(src, dst)

                    warped_bg_rgb = cv2.warpPerspective(
                        bg_rgb_loc, Mbg, (W_img, H_img),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
                    )
                    warped_bg_a = cv2.warpPerspective(
                        bg_a_loc, Mbg, (W_img, H_img),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                    warped_bg_a = np.clip(warped_bg_a, 0, 255).astype(np.uint8)

                    img_bg = self._overlay_alpha_blend(img, warped_bg_rgb, warped_bg_a)
                else:
                    # fallback: старый bbox-rect в image-space
                    img_bg = self._overlay_apply_bg_rect_imgspace(img, warped_a_full, pad_px=16, alpha_thr=bg_thr)
            else:
                img_bg = self._overlay_apply_bg_rect_imgspace(img, warped_a_full, pad_px=16, alpha_thr=bg_thr)

            # blend text (используем альфу ПОСЛЕ окклюзии)
            img_new = self._overlay_alpha_blend(img_bg, warped_rgb, warped_a_vis)

            # --- заполнение зоны окклюдера ---
            if occ_mask_u8 is not None:
                m = (occ_mask_u8 > 0)

                shadow = float(getattr(self, "overlay_occ_shadow_strength", 0.0))
                if shadow > 1e-6:
                    shadow = max(0.0, min(0.6, shadow))
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    ring = cv2.dilate(occ_mask_u8, k, iterations=1)
                    ring = (ring > 0) & (~m)
                    if ring.any():
                        tmp = img_new.astype(np.float32)
                        tmp[ring] = np.clip(tmp[ring] * (1.0 - shadow), 0, 255)
                        img_new = tmp.astype(np.uint8)

                fill_source = str(getattr(self, "overlay_occ_fill_source", "img_bg")).lower()
                fill_img = img if fill_source == "img" else img_bg
                img_new[m] = fill_img[m]
            else:
                if debug:
                    print("[OVERLAY] occ skipped (occ_mask_u8 is None)")

            # charBB
            mask_for_charbb = text_mask_out if return_visible_mask else text_mask_full
            charBB = self._overlay_build_charBB_from_mask(mask_for_charbb, n_chars)

            return img_new, charBB, text_mask_out

        except Exception as e:
            import traceback
            print("[OVERLAY] UNHANDLED EXCEPTION:", repr(e))
            traceback.print_exc()
            return None, None, None
