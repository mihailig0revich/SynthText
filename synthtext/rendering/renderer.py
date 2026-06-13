# Author: Ankush Gupta
# Date: 2015

"""
Main script for synthetic text rendering.
"""

import random

from synthtext.spatial import synth_utils as su
from .colorize import Colorize
from synthtext.debug_viz import init_interactive_matplotlib, to_rgb
from synthtext.spatial.geometry import estimate_local_scale_grid, warp_points as _warp_points
from synthtext.spatial.regions import TextRegions, get_text_placement_mask
from .overlay import RendererOverlayMixin
from .text_service import TextRenderingService
import numpy as np
import cv2

plt = init_interactive_matplotlib()

MIN_FONT_PX   = 14      # минимально допустимая высота шрифта (под себя)
SHRINK_STEP   = 0.90    # шаг уменьшения шрифта при бэкоффе

# Совместимость со старым кодом на NumPy 2.x
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

class RendererV3(RendererOverlayMixin):

    def __init__(self, data_dir, max_time=None):
        self.text_service = TextRenderingService(data_dir)
        self.text_renderer = self.text_service.renderer  # legacy compatibility
        self.colorizer = Colorize(data_dir)

        self.max_time = max_time

        # --- placement / overlap ---
        self.min_box_gap_rect_px = 30  # было 6
        self.min_box_gap_px = 12

        # --- budgets (используются в _compute_budgets/render_text) ---
        self.global_attempt_budget = 40      # максимум попыток текста на инстанс
        self.global_attempt_budget_base = 10
        self.per_region_attempt_cap_base = 3
        self.max_shrink_trials_base = 3

        # runtime (ставится в render_text, но держим поле)
        self._max_shrink_trials_runtime = self.max_shrink_trials_base

        # cache failed (используется в place_text_textfirst)
        self._failed_pairs = set()  # {(ireg:int, f_px:int)}

        # --- text sampling / readability (используются) ---
        self.min_word_len = 4
        self.min_char_px_img = 7

        self.min_text_rel_height = 0.03
        self.min_text_abs_px = 0          # (нужно, читается через getattr)

        self.min_words_per_image = 2      # (нужно, читается через getattr)
        self.max_words_per_image = 4      # (нужно, читается через getattr)

        # --- geometry toggles ---
        self.no_geom = False

        # --- depth-faithful placement geometry ---
        # RANSAC/depth plane should define the fronto-parallel transform.
        # Keep full plane rectification and do not clamp tilt by default.
        self.persp_strength = 1.0
        self.persp_max_tilt_deg = None

        # --- homography / FP debug (используются) ---
        self.debug_hgeom = False
        self.debug_hgeom_max_regions = 3
        self.debug_hgeom_npts = 64
        self.debug_hgeom_print_mats = False

        # --- overlay perspective controls ---
        # Disabled by default: final text projection should follow the
        # depth/RANSAC homography instead of an artificial visual boost.
        self.overlay_min_persp_strength = 0.1
        self.overlay_persp_boost = 0.0
        self.overlay_persp_min_ratio = 0.7
        self.overlay_alpha_thr = 2
        self.overlay_bg_alpha_thr = 12
        self.overlay_canvas_pad_scale = 1.10
        self.overlay_outline_iters = 3

        # если хочешь реально “скипать слабую перспективу” — код смотрит на ЭТО:
        self.overlay_require_perspective = False

        # --- sky ban (используются) ---
        self.overlay_disallow_sky = True
        self.overlay_use_seg_sky_check = True
        self.overlay_sky_labels = {1}
        self.overlay_sky_label_ratio_thr = 0.60

        # эвристика (если seg нет)
        self.overlay_sky_area_thr = 0.18
        self.overlay_sky_y_thr = 0.40
        self.overlay_sky_w_thr = 0.80
        self.overlay_sky_h_thr = 0.25

        # sector-sky (в render_text_overlay читается через getattr — добавляем явные дефолты)
        self.overlay_sky_sector_enable = True
        self.overlay_sky_sector_deg_min = 45.0
        self.overlay_sky_sector_deg_max = 135.0
        self.overlay_sky_sector_votes_thr = 5
        self.overlay_sky_sector_min_r = 0.08
        self.overlay_sky_sector_require_above_center = True

        # --- region selection ---
        self.region_select_topk = 6

        # --- augs master switch (в render_text читается через getattr) ---
        self.disable_all_augs = False

        # --- speed mode (оставляем как было по значениям) ---
        self.fast_mode = True

        # Размер текста внутри сегмента (НЕ на весь сегмент, но и НЕ мелко)
        self.overlay_fill_w_min = 0.45
        self.overlay_fill_w_max = 0.82
        self.overlay_fill_h_min = 0.18
        self.overlay_fill_h_max = 0.38

        # Для очень больших сегментов/канвасов чуть уменьшаем заполнение
        self.overlay_fill_large_thr_px = 700
        self.overlay_fill_large_scale = 0.90

        # Автоподгонка размера
        self.overlay_font_grow_factor = 1.12
        self.overlay_font_grow_iters = 18
        self.overlay_font_grow_ratio_max = 3.0

        self.overlay_region_sector_deg_min = 30.0

        self.overlay_region_sector_deg_max = 150.0

        self.overlay_region_sector_votes_thr = 3 

        self.overlay_region_sector_min_r = 0.03 

        self.overlay_region_sector_require_above_center = True

        # (опционально) минимальный стартовый font.size в оверлее
        # чтобы не начинать с совсем мелкого f_fit
        self.overlay_min_font_start_px = 0

        # --- speed caches / debug switches ---
        self.debug_txt = False          # печать [TXT] логов (сильно тормозит на больших батчах)
        self.debug_overlay = False      # печать [OVERLAY] логов
        self.debug_regions = False      # печать [render_text]/[filter_for_placement] логов

        self._kernel_cache = {}         # k -> cv2 kernel
        self._pygame_inited = False     # pygame init once
        self._overlay_surf_cache = {}   # (Wc,Hc) -> pygame.Surface

        self.min_char_px_img = 12      # было ~8: минимальная "высота символа" в пикселях итогового изображения
        self.min_text_abs_px = 14      # абсолютный минимум высоты текста (px) в изображении
        self.min_text_rel_height = 0.0 # 0.012..0.018 если хочешь завязку на разрешение (например 0.015 для 4K)

        self.max_text_instances = 3

        self.debug = False

        self.log = False

        self.overlay_occ_enable = True
        self.overlay_occ_p = 1.0                    # вероятность окклюзии для данного текста
        self.overlay_occ_cov_range = (0.06, 0.26)    # целевая доля перекрытия текста
        self.overlay_occ_max_cov = 0.42              # жёсткий потолок: не перекрывать больше этого
        self.overlay_occ_n_shapes = (1, 3)           # сколько "объектов" рисуем
        self.overlay_occ_feather_px = (2, 6)         # размытие краёв окклюдера
        self.overlay_occ_opacity = (0.80, 0.98)      # непрозрачность окклюдера
        self.overlay_occ_shift_px = (10, 45)         # насколько смещаем источник текстуры (пикс)
        self.overlay_occ_blur_sigma = (0.0, 1.2)     # defocus у окклюдера
        self.overlay_occ_gain = (0.90, 1.12)         # множитель яркости окклюдера
        self.overlay_occ_gamma = (0.90, 1.15)        # гамма окклюдера

        self.overlay_occ_fill_source = "img"

        self.overlay_occ_p = 0.4                 # чаще
        self.overlay_occ_pieces_range = (2, 5)    # несколько штук
        self.overlay_occ_piece_frac_min = 0.012   # маленькие
        self.overlay_occ_piece_frac_max = 0.045
        self.overlay_occ_min_piece_px = 120       # не "пыль"
        self.overlay_occ_avoid_overlap = True     # чтобы не слипалось

        self.overlay_occ_kind_probs = {"band_poly": 0.25, "sticker": 0.40, "ellipse": 0.25, "edge_block": 0.10}

        self.disable_all_augs = True
        self.noise_mode = "auto"
        self.noise_strength = 1.1   # попробуй 1.20 если хочется пожёстче
        self.noise_p_boost = 1.25

        if self.fast_mode:
            # минимум попыток и ужиманий (как у тебя было)
            self.global_attempt_budget_base = 5
            self.per_region_attempt_cap_base = 2
            self.max_shrink_trials_base = 2
            self._max_shrink_trials_runtime = self.max_shrink_trials_base

            # поменьше «мелочи» — меньше неудачных масок
            self.min_word_len = 5
            self.min_char_px_img = 8




    def filter_for_placement(self, xyz, seg, regions, viz=False):
        """
        Вариант близкий к оригинальному SynthText:
        - для каждого планарного региона считаем fronto-parallel маску через get_text_placement_mask
        - сохраняем place_mask, H, Hinv в словарь regions

        + ДОБАВЛЕНО: логи геометрии (FP маска + H/Hinv sanity)
        """
        import numpy as np

        if regions is None or "label" not in regions or "area" not in regions:
            print("[filter_for_placement] empty regions dict, bail out")
            return None

        n = len(regions["label"])
        print(f"[filter_for_placement] start, n={n}")

        place_masks = []
        homographies = []
        homographies_inv = []
        new_labels = []
        new_areas = []
        new_coeffs = []
        new_rots = []

        labels = np.asarray(regions["label"])
        areas  = np.asarray(regions["area"])

        # >>> ключевой момент: пробуем 'coeff', если нет — 'plane'
        coeffs = regions.get("coeff", None)
        if coeffs is None:
            coeffs = regions.get("plane", None)

        rots = regions.get("rot", None)

        if coeffs is None:
            print("[filter_for_placement] no plane coefficients in regions (no 'coeff' and no 'plane')")
            return None

        coeffs = np.asarray(coeffs)

        # Depth/RANSAC plane controls the rectification geometry.
        persp_strength = float(getattr(self, "persp_strength", 1.0))
        persp_max_tilt_deg = getattr(self, "persp_max_tilt_deg", None)

        # --- debug limits ---
        dbg_on = bool(getattr(self, "debug_hgeom", False))
        dbg_max = int(getattr(self, "debug_hgeom_max_regions", 3))
        dbg_count = 0

        for i in range(n):
            lbl = int(labels[i])

            # маска региона в исходном seg
            mask = (seg == lbl).astype("uint8")

            # коэффициенты плоскости ax + by + cz + d = 0
            if i >= len(coeffs):
                print(f"[filter_for_placement] region {i}, label={lbl}: coeff index out of range, skip")
                continue

            plane = coeffs[i]

            res = get_text_placement_mask(
                xyz, mask, plane,
                pad=2, viz=viz,
                persp_strength=persp_strength,
                max_tilt_deg=persp_max_tilt_deg
            )
            if res is None:
                print(f"[filter_for_placement] region {i}, label={lbl}: get_text_placement_mask -> None (rejected)")
                continue

            place_mask_fp, H, Hinv = res

            if place_mask_fp is None or place_mask_fp.size == 0 or int(place_mask_fp.sum()) < 50:
                print(f"[filter_for_placement] region {i}, label={lbl}: too small rectified mask, skip")
                continue

            # --- LOGS: проверяем FP маску и H/Hinv ---
            if dbg_on and dbg_count < dbg_max:
                try:
                    self._dbg_hgeom(lbl, place_mask_fp, H, Hinv, (seg.shape[0], seg.shape[1]), tag=f" r{i}")
                except Exception as e:
                    print("[HGEOM] debug failed:", repr(e))
                dbg_count += 1

            place_masks.append(place_mask_fp)
            homographies.append(H)
            homographies_inv.append(Hinv)
            new_labels.append(lbl)
            new_areas.append(float(areas[i]))
            new_coeffs.append(plane)
            if rots is not None and len(rots) > i:
                new_rots.append(rots[i])
            else:
                new_rots.append(0.0)

        if len(place_masks) == 0:
            print("[filter_for_placement] all regions rejected -> None")
            return None

        out = {}
        out["label"] = np.asarray(new_labels, dtype=labels.dtype)
        out["area"]  = np.asarray(new_areas,  dtype=areas.dtype)
        out["coeff"] = np.asarray(new_coeffs, dtype=np.float32)
        out["rot"]   = np.asarray(new_rots,   dtype=np.float32)

        out["place_mask"]     = place_masks
        out["homography"]     = homographies
        out["homography_inv"] = homographies_inv

        print(f"[filter_for_placement] done, kept {len(place_masks)} regions")
        return out

    def _ensure_region_cache(self, img, place_masks, regions):
        """
        Один раз на изображение считаем:
        - fp bbox (в FP)
        - quad_img (4x2 TL,TR,BR,BL в image-space) через Hinv
        - bbox_img (x0,y0,x1,y1)
        - s_loc (локальный FP->IMG масштаб)
        - base_angle (угол длинной оси по seg в image-space)
        - score + candidates (сортировка)

        Это сильно сокращает работу на КАЖДОЕ слово.
        """
        import numpy as np

        if img is None or place_masks is None or regions is None:
            self._region_cache = {"candidates": []}
            return

        H_img, W_img = img.shape[:2]
        n = int(len(place_masks))
        if n <= 0:
            self._region_cache = {"candidates": []}
            return

        labels = regions.get("label", None)
        if labels is None or (hasattr(labels, "__len__") and len(labels) != n):
            labels = np.arange(n, dtype=np.int32)

        Hinvs = regions.get("homography_inv", None)
        if Hinvs is None or (hasattr(Hinvs, "__len__") and len(Hinvs) != n):
            Hinvs = [None] * n

        fp_bbox = [(0, 0, 0, 0)] * n
        quad_img = [None] * n
        bbox_img = [(0.0, 0.0, 0.0, 0.0)] * n
        fp_wh = [(0, 0)] * n
        s_loc = [1.0] * n
        base_angle = [0.0] * n
        banned = [False] * n
        score = np.zeros((n,), dtype=np.float64)

        disallow_sky = bool(getattr(self, "overlay_disallow_sky", True))
        area_thr = float(getattr(self, "overlay_sky_area_thr", 0.18))
        y_thr    = float(getattr(self, "overlay_sky_y_thr", 0.40))
        w_thr    = float(getattr(self, "overlay_sky_w_thr", 0.80))
        h_thr    = float(getattr(self, "overlay_sky_h_thr", 0.25))

        seg_last = getattr(self, "_seg_last", None)

        for i in range(n):
            pm = np.asarray(place_masks[i])
            if pm.ndim != 2:
                banned[i] = True
                continue

            ys, xs = np.where(pm == 0)
            if xs.size < 30:
                banned[i] = True
                continue

            x0_fp, x1_fp = int(xs.min()), int(xs.max())
            y0_fp, y1_fp = int(ys.min()), int(ys.max())
            fp_bbox[i] = (x0_fp, y0_fp, x1_fp, y1_fp)
            w_fp = int(x1_fp - x0_fp + 1)
            h_fp = int(y1_fp - y0_fp + 1)
            fp_wh[i] = (w_fp, h_fp)

            Hinv = Hinvs[i]
            # --- quad in image-space (TL,TR,BR,BL) ---
            corners_fp = np.array(
                [[x0_fp, y0_fp], [x1_fp, y0_fp], [x1_fp, y1_fp], [x0_fp, y1_fp]],
                dtype=np.float32
            )

            if Hinv is not None:
                try:
                    q = _warp_points(Hinv, corners_fp)
                    q = self._overlay_order_quad_tl_tr_br_bl(q)
                    quad_img[i] = q.astype(np.float32)
                except Exception:
                    quad_img[i] = corners_fp.astype(np.float32)
            else:
                quad_img[i] = corners_fp.astype(np.float32)

            q = np.asarray(quad_img[i], dtype=np.float32).reshape(4, 2)
            x0 = float(np.clip(np.min(q[:, 0]), 0.0, float(W_img - 1)))
            x1 = float(np.clip(np.max(q[:, 0]), 0.0, float(W_img - 1)))
            y0 = float(np.clip(np.min(q[:, 1]), 0.0, float(H_img - 1)))
            y1 = float(np.clip(np.max(q[:, 1]), 0.0, float(H_img - 1)))
            if x1 < x0: x0, x1 = x1, x0
            if y1 < y0: y0, y1 = y1, y0
            bbox_img[i] = (x0, y0, x1, y1)

            # --- s_loc (один раз!) ---
            try:
                free_mask_fp = (pm == 0).astype(np.uint8)
                # чуть меньше выборок -> быстрее, но достаточно стабильно
                s = float(estimate_local_scale_grid(Hinv, free_mask_fp, k=5, delta=6))
                if (not np.isfinite(s)) or s <= 1e-6:
                    s = 1.0
            except Exception:
                s = 1.0
            s_loc[i] = float(s)

            # --- base_angle (из seg, быстрее/стабильнее) ---
            try:
                lbl = int(labels[i])
                if seg_last is not None:
                    ang = float(self.estimate_region_angle_from_seg(seg_last, lbl))
                else:
                    ang = 0.0
                base_angle[i] = float(self._clamp_readable_angle(ang))
            except Exception:
                base_angle[i] = 0.0

            # --- cheap sky-like ban (bbox heuristic) ---
            if disallow_sky:
                bw = (x1 - x0 + 1.0) / max(1.0, float(W_img))
                bh = (y1 - y0 + 1.0) / max(1.0, float(H_img))
                yc = (0.5 * (y0 + y1)) / max(1.0, float(H_img))
                a  = ((x1 - x0 + 1.0) * (y1 - y0 + 1.0)) / max(1.0, float(H_img * W_img))
                if (a >= area_thr) and (yc <= y_thr) and (bw >= w_thr) and (bh >= h_thr):
                    banned[i] = True

            # --- score: крупнее + ниже + нормированный scale ---
            area_img = max(1.0, (x1 - x0 + 1.0) * (y1 - y0 + 1.0))
            yc = (0.5 * (y0 + y1)) / max(1.0, float(H_img))
            w_y = 0.70 + 0.60 * float(np.clip(yc, 0.0, 1.0))
            w_s = float(np.clip(s, 0.6, 2.5))
            sc = area_img * w_y * w_s
            score[i] = 0.0 if banned[i] else float(sc)

        cand = np.argsort(-score).astype(np.int32).tolist()
        self._region_cache = {
            "score": score,
            "fp_bbox": fp_bbox,
            "fp_wh": fp_wh,
            "quad_img": quad_img,
            "bbox_img": bbox_img,
            "s_loc": s_loc,
            "base_angle": base_angle,
            "banned": banned,
            "candidates": cand,
        }



    def select_region_for_text(self, txt, font, f_layout, f_asp, place_masks, regions,
        gap_px=6, min_font_px=14, shrink_step=0.90, side_margin=0.90,
        min_text_px_img=80, occupied_global=None, fast_mode=True,
        img=None, nline=1, nchar=10, force_ireg=None, **kwargs):
        """
        ВАЖНОЕ ИЗМЕНЕНИЕ:
        - Раньше регион добавлялся в self._used_regions_this_image прямо тут на "accept".
        Это ломало повторные попытки в одном и том же force_ireg (первая попытка упала → регион уже used).
        - Теперь used помечается ТОЛЬКО ПОСЛЕ УСПЕШНОГО overlay в place_text_textfirst().
        """
        import numpy as np

        debug_txt = bool(getattr(self, "debug_txt", False))
        debug_regions = bool(getattr(self, "debug_regions", False))
        dbg_on = debug_txt or debug_regions

        def _dbg(msg, **kw):
            if not dbg_on:
                return
            s = f"[TXT] select_region_for_text: {msg}"
            if kw:
                s += " | " + ", ".join(f"{k}={v}" for k, v in kw.items())
            print(s)

        if "min_char_px_img" in kwargs:
            try:
                min_text_px_img = int(kwargs.get("min_char_px_img"))
            except Exception:
                pass

        if img is None:
            _dbg("fail: img is None")
            return None, None, None

        H_img, W_img = img.shape[:2]
        self._img_shape_last = (H_img, W_img)

        if (not hasattr(self, "_region_cache")) or (self._region_cache is None) or (not self._region_cache.get("candidates")):
            self._ensure_region_cache(img, place_masks, regions)

        cache = self._region_cache
        if not cache or not cache.get("candidates"):
            _dbg("fail: region_cache empty")
            return None, None, None

        topk = int(getattr(self, "region_select_topk", 6))
        tries = int(getattr(self, "region_select_tries", 6))
        fill = float(getattr(self, "text_fill_factor", 0.70))
        angle_jitter = float(getattr(self, "angle_jitter_deg", 7.0))

        avoid_repeat = bool(getattr(self, "avoid_repeat_region", True))
        if avoid_repeat and (not hasattr(self, "_used_regions_this_image") or self._used_regions_this_image is None):
            self._used_regions_this_image = set()
        used = self._used_regions_this_image if avoid_repeat else set()

        def _occupied_ok(b):
            if occupied_global is None:
                return True
            try:
                x0, y0, x1, y1 = b
                x0 = int(max(0, min(W_img - 1, x0)))
                x1 = int(max(0, min(W_img - 1, x1)))
                y0 = int(max(0, min(H_img - 1, y0)))
                y1 = int(max(0, min(H_img - 1, y1)))
                if x1 <= x0 or y1 <= y0:
                    return True
                roi = occupied_global[y0:y1, x0:x1]
                occ = float((roi > 0).mean())
                ok = occ < float(getattr(self, "occupied_bbox_max_frac", 0.15))
                return ok
            except Exception:
                return True

        def _try_region(i: int):
            if i is None:
                return None, None, None
            i = int(i)

            if i < 0 or i >= len(cache["banned"]):
                _dbg("reject: idx out of range", i=i, n=len(cache["banned"]))
                return None, None, None
            if bool(cache["banned"][i]):
                _dbg("reject: banned", i=i)
                return None, None, None
            if avoid_repeat and (i in used):
                _dbg("reject: already used", i=i)
                return None, None, None
            if not _occupied_ok(cache["bbox_img"][i]):
                _dbg("reject: occupied_global", i=i, bbox=cache["bbox_img"][i])
                return None, None, None

            fp_w, fp_h = cache["fp_wh"][i]
            fp_w = float(fp_w); fp_h = float(fp_h)
            if fp_w <= 0 or fp_h <= 0:
                _dbg("reject: bad fp_wh", i=i, fp_w=fp_w, fp_h=fp_h)
                return None, None, None

            s = float(cache["s_loc"][i])
            if (not np.isfinite(s)) or s <= 1e-6:
                s = 1.0

            nchar_eff = max(3, int(nchar))
            nline_eff = max(1, int(nline))

            denom_w = (nchar_eff * float(f_asp) + 0.15 * (nchar_eff - 1))
            f_max_w = (fp_w * float(side_margin)) / max(denom_w, 1e-6)

            line_h = 1.15
            f_max_h = (fp_h * float(side_margin)) / max((nline_eff * line_h), 1e-6)

            f_max = float(min(f_max_w, f_max_h))
            if (not np.isfinite(f_max)) or (f_max < float(min_font_px)):
                _dbg("reject: f_max too small", i=i, f_max=round(f_max, 2), min_font_px=min_font_px,
                    fp_w=int(fp_w), fp_h=int(fp_h), nchar=nchar_eff, nline=nline_eff, f_asp=round(float(f_asp), 3))
                return None, None, None

            f_nom = f_max * float(fill)
            f_min_req = float(min_text_px_img) / max(1e-6, s)
            f_fit = float(max(f_nom, f_min_req, float(min_font_px)))
            if f_fit > f_max:
                _dbg("reject: f_fit > f_max", i=i, f_fit=round(f_fit, 2), f_max=round(f_max, 2),
                    min_text_px_img=min_text_px_img, s_loc=round(float(s), 3))
                return None, None, None

            base_ang = float(cache["base_angle"][i])
            selected_angle = base_ang + float(np.random.uniform(-angle_jitter, angle_jitter))

            _dbg("accept", i=i, f_fit=round(f_fit, 2), base_ang=round(base_ang, 1),
                sel_ang=round(selected_angle, 1), s_loc=round(float(s), 3))
            return i, f_fit, selected_angle

        if force_ireg is not None:
            _dbg("force_ireg", force_ireg=int(force_ireg))
            return _try_region(int(force_ireg))

        cand = cache["candidates"]
        pick_pool = cand[:max(1, min(topk, len(cand)))]

        scores = np.array([cache["score"][i] for i in pick_pool], dtype=np.float64)
        scores = np.maximum(scores, 1e-9)
        probs = scores / scores.sum()

        for t in range(int(tries)):
            i = int(np.random.choice(pick_pool, p=probs))
            out = _try_region(i)
            if out[0] is not None:
                return out
            _dbg("try failed", t=t, picked=i)

        _dbg("fail: no region matched", tries=tries, topk=topk, pool=len(pick_pool))
        return None, None, None



    def place_text_textfirst(self, img, place_masks, regions, gap=6,
        min_font_px=14, start_font_px=None, start_font_px_range=None,
        shrink_step=0.90, depth=None, occupied_global=None, force_ireg=None):
        import numpy as np

        debug_txt = bool(getattr(self, "debug_txt", False))

        def _dbg(msg, **kw):
            if not debug_txt:
                return
            s = f"[TXT] place_text_textfirst: {msg}"
            if kw:
                s += " | " + ", ".join(f"{k}={v}" for k, v in kw.items())
            print(s)

        if img is None:
            _dbg("fail: img is None")
            return None

        H_img, W_img = img.shape[:2]
        self._img_shape_last = (H_img, W_img)

        # place_masks может быть list; если numpy-массив, "if not place_masks" может падать
        try:
            if place_masks is None or len(place_masks) == 0:
                _dbg("fail: empty place_masks")
                return None
        except Exception:
            _dbg("fail: place_masks len() failed")
            return None

        # ✅ per-image failed pairs (НЕ глобальные)
        if not hasattr(self, "_failed_pairs_this_image") or self._failed_pairs_this_image is None:
            self._failed_pairs_this_image = set()
        if not hasattr(self, "_failed_pair_counts_this_image") or self._failed_pair_counts_this_image is None:
            self._failed_pair_counts_this_image = {}

        failed_set = self._failed_pairs_this_image
        failed_cnt = self._failed_pair_counts_this_image
        ban_after = int(getattr(self, "failed_pair_ban_after", 3))

        # --- font init ---
        try:
            font_ctx = self.text_service.sample_font()
            font = font_ctx.font
            f_asp = float(font_ctx.aspect_ratio)
        except Exception as e:
            _dbg("fail: font init", err=repr(e))
            return None

        short_side = float(min(H_img, W_img))
        base_char = max(float(min_font_px), short_side / 14.0)

        # --- стартовый размер ---
        try:
            if start_font_px_range is not None:
                lo_rng, hi_rng = start_font_px_range
                if lo_rng > hi_rng:
                    lo_rng, hi_rng = hi_rng, lo_rng
                f_start = int(np.random.randint(int(lo_rng), int(hi_rng) + 1))
                f_start = max(f_start, int(min_font_px))
                src = "range"
            elif start_font_px is not None:
                f_start = max(float(start_font_px), float(min_font_px))
                src = "explicit"
            else:
                last = getattr(self, "last_font_h_px", None)
                if last is not None:
                    f_start = max(float(last), base_char)
                    src = "last_font_h_px"
                else:
                    f_start = base_char
                    src = "base_char"

            jitter = float(np.random.uniform(0.9, 1.1))
            f_start = float(np.clip(f_start * jitter, float(min_font_px), short_side / 16.0))
            f_layout = int(round(f_start * 1.30))
            if f_layout < int(min_font_px):
                f_layout = int(min_font_px)

            _dbg("font sizing", source=src, f_start=round(f_start, 2), f_layout=f_layout, f_asp=round(float(f_asp), 3))
        except Exception as e:
            _dbg("fail: font sizing", err=repr(e))
            return None

        # --- layout ---
        nline_eff, nchar_eff = self.text_service.estimate_layout_capacity(
            f_layout,
            f_asp,
            mask_size=(128, 512),
            min_chars=6,
            fallback=(1, 12),
        )

        # --- text sample (NEW: получаем язык) ---
        try:
            txt_str, txt_lang = self._sample_layout_text(nline_eff, nchar_eff, max_retries=5)
        except Exception as e:
            _dbg("fail: _sample_layout_text exception", err=repr(e))
            return None

        if not txt_str or not isinstance(txt_str, str) or not txt_str.strip():
            _dbg("fail: sampled empty text", txt=repr(txt_str))
            return None
        txt_str = txt_str.strip()
        if not txt_lang:
            txt_lang = "unk"

        # --- ensure cache ---
        if (not hasattr(self, "_region_cache")) or (self._region_cache is None) or (not self._region_cache.get("candidates")):
            self._ensure_region_cache(img, place_masks, regions)
        cache = self._region_cache
        if not cache or not cache.get("candidates"):
            _dbg("fail: region cache empty after build")
            return None

        # --- select region ---
        ireg, f_fit, selected_angle = self.select_region_for_text(
            txt_str, font, f_layout, f_asp, place_masks, regions,
            gap_px=gap,
            min_font_px=min_font_px,
            shrink_step=shrink_step,
            side_margin=0.92,
            min_char_px_img=int(getattr(self, "min_char_px_img", 8)),
            fast_mode=getattr(self, "fast_mode", False),
            img=img,
            nline=nline_eff,
            nchar=nchar_eff,
            occupied_global=occupied_global,
            force_ireg=force_ireg
        )
        if ireg is None:
            _dbg("fail: select_region_for_text returned None", force_ireg=force_ireg, txt=txt_str)
            return None

        region_coords = cache["quad_img"][ireg]
        if region_coords is None:
            _dbg("fail: cache quad_img is None", ireg=ireg)
            return None

        (x0, y0, x1, y1) = cache["bbox_img"][ireg]
        near_w = max(2.0, float(x1 - x0))
        near_h = max(2.0, float(y1 - y0))
        s_loc = float(cache["s_loc"][ireg]) if cache.get("s_loc") is not None else 1.0
        if (not np.isfinite(s_loc)) or s_loc <= 1e-6:
            s_loc = 1.0

        # --- min readable px ---
        min_text_px_img = int(getattr(self, "min_char_px_img", 8))
        rel = float(getattr(self, "min_text_rel_height", 0.0))
        rel_px = int(round(rel * float(min(H_img, W_img))))
        abs_px = int(getattr(self, "min_text_abs_px", 0))
        min_text_px_img = max(min_text_px_img, rel_px, abs_px)

        # --- визуальный подбор размера ---
        fill = float(getattr(self, "overlay_text_fill", 0.65))
        fill = max(0.30, min(0.92, fill))

        max_char_frac_h = float(getattr(self, "overlay_max_char_frac_h", 0.72))
        max_char_frac_h = max(0.40, min(0.90, max_char_frac_h))

        char_h_by_w = (fill * near_w) / (max(1.0, float(nchar_eff)) * max(1e-6, float(f_asp)) * 1.08)
        char_h_by_h = (fill * near_h) / (max(1.0, float(nline_eff)) * 1.25)

        char_h_px = float(min(char_h_by_w, char_h_by_h))
        char_h_px = max(float(min_text_px_img), char_h_px)
        char_h_px = min(char_h_px, max_char_frac_h * float(near_h))

        f_target_fp = float(char_h_px) / max(1e-6, float(s_loc))
        f_max_fp = max(float(min_font_px), float(f_layout) * 2.2)
        f_target_fp = float(np.clip(f_target_fp, float(min_font_px), float(f_max_fp)))

        (x0fp, y0fp, x1fp, y1fp) = cache["fp_bbox"][ireg]
        W_reg = float(max(2, int(x1fp - x0fp + 1)))
        H_reg = float(max(2, int(y1fp - y0fp + 1)))
        side_margin = 0.92
        gap_px = int(gap)

        def fits_fp(f_fp: float) -> bool:
            ch = float(f_fp)
            cw = float(f_fp) * float(f_asp)
            total_h = float(nline_eff) * ch * 1.30 + 2.0 * gap_px
            total_w = float(nchar_eff) * cw * 1.10 + 2.0 * gap_px
            return (total_w <= W_reg * side_margin) and (total_h <= H_reg * side_margin)

        f_final = float(f_target_fp)
        shrink = float(getattr(self, "overlay_visual_shrink", 0.93))
        shrink = max(0.85, min(0.98, shrink))
        tries = 0
        while tries < 12 and (not fits_fp(f_final)) and f_final > float(min_font_px) + 1e-3:
            f_final *= shrink
            tries += 1

        if fits_fp(float(f_fit)) and f_final < float(f_fit):
            f_final = float(f_fit)

        _dbg("visual sizing",
            ireg=ireg, txt=txt_str, nchar=nchar_eff, nline=nline_eff,
            s_loc=round(float(s_loc), 3), near_w=round(float(near_w), 1), near_h=round(float(near_h), 1),
            fill=round(float(fill), 2), char_h_px=round(float(char_h_px), 1),
            f_fit=round(float(f_fit), 1), f_target_fp=round(float(f_target_fp), 1),
            f_final=round(float(f_final), 1), tries_shrink=tries, force_ireg=force_ireg
        )

        # ✅ ключ делаем чуть устойчивее (не только ireg+font)
        key = (int(ireg), int(round(float(f_final))), int(nchar_eff), int(nline_eff))

        # если ключ уже забанен — пробуем "чуть меньше" (другой ключ), а не сразу отваливаемся
        if key in failed_set:
            f2 = max(float(min_font_px), float(f_final) * 0.92)
            key2 = (int(ireg), int(round(float(f2))), int(nchar_eff), int(nline_eff))
            if key2 not in failed_set:
                _dbg("failed_pairs: key banned -> try smaller f_final", key=key, key2=key2)
                f_final = f2
                key = key2
            else:
                _dbg("fail: in failed_pairs cache", key=key)
                return None

        def _bump_fail(reason: str):
            c = int(failed_cnt.get(key, 0)) + 1
            failed_cnt[key] = c
            banned = False
            if c >= ban_after:
                failed_set.add(key)
                banned = True
            _dbg("pair fail", key=key, reason=reason, count=c, ban_after=ban_after, banned=banned)

        # set font.size
        try:
            self.text_service.set_font_size_px(font, float(f_final))
        except Exception as e:
            _dbg("fail: set font.size", err=repr(e), f_final=round(float(f_final), 2))
            _bump_fail("set_font_size")
            return None

        # render overlay (с fallback по sky-ban)
        def _call_overlay():
            return self.render_text_overlay(
                img, txt_str, font,
                selected_angle=selected_angle,
                region_coords=region_coords,
                depth=depth
            )

        try:
            img_new, bb_img, text_mask_img = _call_overlay()
        except Exception as e:
            _dbg("fail: render_text_overlay exception", err=repr(e))
            _bump_fail("overlay_exc")
            return None

        # fallback: если отвалилось, а bbox точно НЕ в верхней части кадра — вероятно sky-ban ложноположительный
        if (img_new is None or text_mask_img is None):
            disallow_sky = bool(getattr(self, "overlay_disallow_sky", True))
            if disallow_sky:
                y_center = (float(y0) + float(y1)) * 0.5 / max(1.0, float(H_img))
                y_thr = float(getattr(self, "overlay_sky_fallback_min_y", 0.30))
                if y_center >= y_thr:
                    _dbg("overlay fallback: disable sky-ban once", y_center=round(y_center, 3), thr=round(y_thr, 3))
                    old = getattr(self, "overlay_disallow_sky", True)
                    try:
                        self.overlay_disallow_sky = False
                        img_new2, bb_img2, text_mask_img2 = _call_overlay()
                        if img_new2 is not None and text_mask_img2 is not None:
                            img_new, bb_img, text_mask_img = img_new2, bb_img2, text_mask_img2
                    finally:
                        self.overlay_disallow_sky = old

        if img_new is None or text_mask_img is None:
            _dbg("fail: render_text_overlay returned None",
                img_new_is_none=(img_new is None), mask_is_none=(text_mask_img is None))
            _bump_fail("overlay_none")
            return None

        nz = int((text_mask_img > 0).sum())
        if nz == 0:
            _dbg("fail: text_mask_img empty after overlay", nz=nz)
            _bump_fail("mask_empty")
            return None

        # ✅ помечаем used (только после успеха!)
        try:
            if bool(getattr(self, "avoid_repeat_region", True)):
                if not hasattr(self, "_used_regions_this_image") or self._used_regions_this_image is None:
                    self._used_regions_this_image = set()
                self._used_regions_this_image.add(int(ireg))
        except Exception:
            pass

        self.last_font_h_px = float(f_final)
        _dbg("success", ireg=ireg, key=key, mask_nz=nz, lang=txt_lang)

        # NEW: возвращаем язык 5-м элементом
        return img_new, txt_str, bb_img, text_mask_img, txt_lang







    def get_num_text_regions(self, nregions: int) -> int:
        import numpy as np

        if nregions <= 0:
            return 0

        min_blocks = int(getattr(self, "min_words_per_image", 2))
        max_blocks = int(getattr(self, "max_words_per_image", 4))
        if max_blocks < min_blocks:
            max_blocks = min_blocks

        # ЖЁСТКИЙ КЭП на 4 текста (или self.max_text_instances)
        max_texts = int(getattr(self, "max_text_instances", 4))
        max_blocks = min(max_blocks, max_texts)

        # и, конечно, не больше чем регионов
        max_blocks = min(max_blocks, int(nregions))
        min_blocks = min(min_blocks, max_blocks)

        if max_blocks <= 0:
            return 0

        k = int(np.random.randint(min_blocks, max_blocks + 1))
        return max(1, min(k, int(nregions)))



    def char2wordBB(self, charBB, text,
                pad_px=0, pad_rel=0.0, clamp_shape=None):
    
        import numpy as np, cv2, itertools, math

        wrds = text.split()
        bb_idx = np.r_[0, np.cumsum([len(w) for w in wrds])]
        m = len(wrds)
        wordBB = np.zeros((2, 4, m), dtype='float32')
        
        if charBB.size == 0 or m == 0:
            return wordBB

        for i in range(m):
            # Собрать все углы символов слова -> (4*n_i) x 2
            cc = charBB[:, :, bb_idx[i]:bb_idx[i+1]]
            cc = np.squeeze(np.concatenate(np.dsplit(cc, cc.shape[-1]), axis=1)).T.astype('float32')

            # Минимальный повернутый прямоугольник
            rect = cv2.minAreaRect(cc.copy())  # ((cx, cy), (w, h), angle_deg)
            (cx, cy), (w, h), angle = rect

            # Увеличение размеров на паддинги
            w_pad = max(1.0, float(w * (1.0 + 2.0 * pad_rel) + 2.0 * pad_px))
            h_pad = max(1.0, float(h * (1.0 + 2.0 * pad_rel) + 2.0 * pad_px))

            # Собрать новый прямоугольник с тем же углом, но без смещения
            rect_padded = ((cx, cy), (w_pad, h_pad), angle)  # Центр остается неизменным
            box = cv2.boxPoints(rect_padded)  # 4x2

            # Согласовать порядок вершин как в исходной реализации
            cc_tblr = np.c_[cc[0, :], cc[-3, :], cc[-2, :], cc[3, :]].T
            perm4 = np.array(list(itertools.permutations(np.arange(4))))
            dists = [np.sum(np.linalg.norm(box[p] - cc_tblr, axis=1)) for p in perm4]
            box = box[perm4[int(np.argmin(dists))], :]

            wordBB[:, :, i] = box.T

        return wordBB



    def render_text(self, rgb, depth, seg, area, label, ninstance=1, viz=False):
        import random
        import numpy as np
        import cv2

        debug_regions = bool(getattr(self, "debug_regions", False))

        def _dbg(msg, **kw):
            if not debug_regions:
                return
            s = f"[render_text] {msg}"
            if kw:
                s += " | " + ", ".join(f"{k}={v}" for k, v in kw.items())
            print(s)

        try:
            depth = np.asarray(depth)
            rgb = np.asarray(rgb)
            seg = np.asarray(seg)
        except Exception as e:
            _dbg("fail: input to np.asarray", err=repr(e))
            return []

        depth_f = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        seg_i = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32, copy=False)
        rgb = to_rgb(rgb)

        # per-image state
        self._seg_last = seg_i
        self._cur_seg = seg_i
        self._img_shape_last = rgb.shape[:2]
        self._region_cache = None
        self._region_cache_key = None

        disable_augs = bool(getattr(self, "disable_all_augs", True))

        try:
            _dbg("start", rgb_shape=tuple(rgb.shape), seg_shape=tuple(seg_i.shape), ninstance=int(ninstance))

            xyz = su.DepthCamera.depth2xyz(depth_f)

            regions = TextRegions.get_regions(xyz, seg_i, area, label)
            _dbg("get_regions", n=int(len(regions.get("label", []))))

            regions = TextRegions.filter_depth(xyz, seg_i, regions)
            _dbg("filter_depth", n=int(len(regions.get("label", []))))

            regions = self.filter_for_placement(xyz, seg_i, regions, viz=False)
            if regions is None:
                _dbg("stop: filter_for_placement returned None")
                return []
            if len(regions.get("place_mask", [])) == 0:
                _dbg("stop: filter_for_placement has 0 place_mask")
                return []

            nregions = len(regions["place_mask"])
            if nregions < 1:
                _dbg("stop: nregions < 1")
                return []

            max_texts = int(getattr(self, "max_text_instances", 4))

            target_blocks = self.get_num_text_regions(nregions)
            target_blocks = int(max(1, min(target_blocks, max_texts)))

            global_budget, per_region_cap, max_shrink_trials = self._compute_budgets(nregions, target_blocks)
            global_budget = max(int(global_budget), int(target_blocks) * 5)

            self.global_attempt_budget = int(global_budget)
            self.per_region_attempt_cap = int(per_region_cap)
            self._max_shrink_trials_runtime = int(max_shrink_trials)

            self._ensure_region_cache(rgb, regions["place_mask"], regions)

            _dbg("budgets", nregions=nregions, target_blocks=target_blocks,
                global_budget=int(self.global_attempt_budget),
                per_region_cap=int(self.per_region_attempt_cap))
        except Exception as e:
            _dbg("region prep error", err=repr(e))
            return []

        res = []

        for inst in range(int(ninstance)):
            img = rgb.copy()
            itext, ibb, ilangs = [], [], []
            occupied_global = np.zeros(img.shape[:2], dtype=np.uint8)

            # ✅ per-output-image state (важно!)
            self._used_regions_this_image = set()
            self._failed_pairs_this_image = set()
            self._failed_pair_counts_this_image = {}

            nregions = len(regions["place_mask"])
            if nregions <= 0:
                _dbg("instance skip: nregions<=0", inst=inst)
                continue

            max_texts = int(getattr(self, "max_text_instances", 4))
            target_blocks = int(max(1, min(self.get_num_text_regions(nregions), max_texts)))

            region_order = list(range(nregions))
            random.shuffle(region_order)

            tries_left = int(getattr(self, "global_attempt_budget", 10))
            per_region_cap = int(getattr(self, "per_region_attempt_cap", 2))
            placed_count = 0

            gap_px = int(getattr(self, "min_box_gap_px", 12))
            ksz = int(2 * gap_px + 1)
            ker = self._get_cached_kernel(ksz)

            stats = {"place_none": 0, "overlay_none": 0, "mask_empty": 0, "overlap": 0, "placed": 0}

            _dbg("instance start",
                inst=inst, target_blocks=target_blocks, nregions=nregions,
                tries_left=tries_left, per_region_cap=per_region_cap)

            for ireg in region_order:
                if placed_count >= target_blocks or tries_left <= 0:
                    break

                for _t in range(per_region_cap):
                    if placed_count >= target_blocks or tries_left <= 0:
                        break
                    tries_left -= 1

                    txt_render_res = self.place_text_textfirst(
                        img,
                        place_masks=regions["place_mask"],
                        regions=regions,
                        gap=int(getattr(self, "min_box_gap_rect_px", 8)),
                        min_font_px=MIN_FONT_PX,
                        shrink_step=SHRINK_STEP,
                        depth=depth_f,
                        occupied_global=occupied_global,
                        force_ireg=int(ireg)
                    )
                    if txt_render_res is None:
                        stats["place_none"] += 1
                        continue

                    # NEW: ожидаем 5 значений
                    try:
                        img_new, text, bb, warped_mask, txt_lang = txt_render_res
                    except Exception:
                        stats["place_none"] += 1
                        continue

                    if not txt_lang:
                        txt_lang = "unk"

                    if img_new is None or warped_mask is None:
                        stats["overlay_none"] += 1
                        continue

                    m_img = (warped_mask > 0).astype(np.uint8) * 255
                    if int(m_img.sum()) == 0:
                        stats["mask_empty"] += 1
                        continue

                    overlap = cv2.bitwise_and(occupied_global, m_img)
                    if int(overlap.sum()) > 0:
                        stats["overlap"] += 1
                        continue

                    m_inflated = cv2.dilate(m_img, ker, 1)
                    occupied_global = np.maximum(occupied_global, m_inflated)

                    img = img_new
                    itext.append(text)
                    ilangs.append(txt_lang)
                    ibb.append(bb)
                    placed_count += 1
                    stats["placed"] += 1

                    _dbg("placed",
                        inst=inst, ireg=int(ireg),
                        placed_count=placed_count, target_blocks=target_blocks,
                        tries_left=tries_left, text=text, lang=txt_lang)
                    break  # 1 текст на регион

            if placed_count == 0:
                _dbg("instance result: placed_count==0",
                    inst=inst, target_blocks=target_blocks,
                    tries_left=tries_left, stats=stats)
                continue

            _dbg("instance result: success",
                inst=inst, placed_count=placed_count, target_blocks=target_blocks,
                tries_left=tries_left, stats=stats)

            idict = {'img': img, 'txt': itext, 'lang': ilangs, 'charBB': None, 'wordBB': None}

            bbs_valid = [b for b in ibb if b is not None and hasattr(b, "shape") and b.size > 0]
            if bbs_valid:
                try:
                    idict['charBB'] = np.concatenate(bbs_valid, axis=2)
                except Exception as e:
                    _dbg("warn: concat charBB failed", err=repr(e), n=len(bbs_valid))
                    idict['charBB'] = None

            if idict['charBB'] is not None:
                try:
                    H, W = img.shape[:2]
                    idict['wordBB'] = self.char2wordBB(
                        idict['charBB'].copy(),
                        ' '.join(itext),
                        pad_px=4, pad_rel=0.05, clamp_shape=(H, W)
                    )
                except Exception as e:
                    _dbg("warn: char2wordBB failed", err=repr(e))
                    idict['wordBB'] = None

            if not disable_augs:
                try:
                    from synthtext.augmentation.noise import apply_noise_recipe
                    cfg = getattr(self, "noise_cfg", None)

                    p_none = float(getattr(self, "noise_p_none", 0.12))
                    p_boost = float(getattr(self, "noise_p_boost", 1.0))
                    strength = float(getattr(self, "noise_strength", 1.0))
                    force_one = bool(getattr(self, "noise_force_one", False))

                    idict["img"], applied = apply_noise_recipe(
                        idict["img"],
                        cfg=cfg,
                        p_none=p_none,
                        p_boost=p_boost,
                        strength=strength,
                        force_at_least_one=force_one,
                    )
                    _dbg("augs applied", inst=inst, applied=applied)
                except Exception as e:
                    _dbg("warn: augs failed", inst=inst, err=repr(e))

            res.append(idict)

        _dbg("done", out_n=int(len(res)))
        return res
