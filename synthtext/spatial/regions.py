"""Scene-region filtering and fronto-parallel placement masks."""

import cv2
import numpy as np

from . import synth_utils as su
from .geometry import rescale_frontoparallel, rot3d_scaled


class TextRegions(object):
    """
    Get region from segmentation which are good for placing
    text.
    """
    minAspect = 0.3  # w > 0.3*h
    maxAspect = 7
    minArea = 100  # number of pix
    minWidth = 24      # было 30
    minHeight = 24     # было 30
    pArea = 0.55

    # RANSAC planar fitting params:
    dist_thresh = 0.30        # было 0.20/0.10 — допускаем дальше от плоскости
    num_inlier = 25           # было 40/90  — требуем меньше инлаеров
    ransac_fit_trials = 80    # немного больше итераций, чтобы что-то нашёл
    ransac_nsample = 20       # сколько точек брать на одну гипотезу плоскости
    min_z_projection = 0.05   # было 0.15 — почти любую «фронтальную» нормаль пускаем
    inlier_ratio = 0.08       # мягкий порог: реальные depth-регионы часто дают 8-10%

    minW = 16

    # <<< НОВОЕ: ограничиваем число регионов, куда полезем с RANSAC и плоскостями >>>
    maxRegionsForPlaneFit = 15  # максимум регионов после TextRegions.filter
    maxPlaneTrials = 15         # максимум регионов в TextRegions.filter_depth
    region_workers = 1          # >1 включает параллельную проверку кандидатов в filter_depth
    ransac_debug = False
    stats_collector = None

    skip_sky_like = True

    sky_min_area_frac = 0.18     # доля пикселей кадра
    sky_max_y_center  = 0.38     # центр bbox по Y должен быть вверху
    sky_min_w_frac    = 0.45     # bbox широкий
    sky_max_h_frac    = 0.75     # bbox не должен быть слишком высокий

    @staticmethod
    def filter_rectified(mask):
        """
        mask : 1 where "ON", 0 where "OFF"
        """
        wx = float(np.median(np.sum(mask, axis=0)))
        wy = float(np.median(np.sum(mask, axis=1)))
        # позволим более узкие длинные области
        return (wx > TextRegions.minW * 0.8) and (wy > TextRegions.minW * 0.6)

    @staticmethod
    def get_hw(pt, return_rot=False):
        pt = pt.copy()
        R = su.unrotate2d(pt)
        mu = np.median(pt, axis=0)
        pt = (pt - mu[None, :]).dot(R.T) + mu[None, :]
        h, w = np.max(pt, axis=0) - np.min(pt, axis=0)
        if return_rot:
            return h, w, R
        return h, w

    @staticmethod
    def filter(seg, area, label):
        """
        Apply the filter.
        The final list is ranked by area.
        """
        stats_collector = getattr(TextRegions, "stats_collector", None)

        def _record(event, **fields):
            if stats_collector is None:
                return
            try:
                item = {"event": event}
                item.update(fields)
                stats_collector.append(item)
            except Exception:
                pass

        label = np.asarray(label)
        area = np.asarray(area)
        area_keep = area > TextRegions.minArea
        n_small_area = int(np.sum(~area_keep))
        if n_small_area:
            _record("shape_rejected_small_area", count=n_small_area, min_area=float(TextRegions.minArea))

        good = label[area_keep]
        area = area[area_keep]
        filt, R = [], []
        for idx, i in enumerate(good):
            mask = (seg == i)

            # np.where -> (rows, cols) = (y, x)
            ys, xs = np.where(mask)

            # OpenCV ждёт точки (x, y)
            coords = np.c_[xs, ys].astype('float32')

            if coords.shape[0] < 10:
                _record("shape_rejected_too_few_points", label=int(i), points=int(coords.shape[0]))
                filt.append(False)
                R.append(np.eye(2, dtype=np.float32))
                continue

            rect = cv2.minAreaRect(coords)
            box = np.array(cv2.boxPoints(rect))
            h, w, rot = TextRegions.get_hw(box, return_rot=True)

            # --- мягкий фильтр по аспект-ратио ---
            aspect = max(float(h) / max(float(w), 1.0),
                        float(w) / max(float(h), 1.0))

            rect_area = max(1.0, float(w) * float(h))
            f = (
                h > TextRegions.minHeight * 0.8 and
                w > TextRegions.minWidth * 0.8 and
                (float(area[idx]) / rect_area) >= (TextRegions.pArea * 0.85) and
                aspect < 18.0
            )
            if f:
                _record("shape_accepted", label=int(i), area=float(area[idx]), h=float(h), w=float(w), aspect=aspect)
            elif h <= TextRegions.minHeight * 0.8 or w <= TextRegions.minWidth * 0.8:
                _record("shape_rejected_small_bbox", label=int(i), area=float(area[idx]), h=float(h), w=float(w))
            elif (float(area[idx]) / rect_area) < (TextRegions.pArea * 0.85):
                _record(
                    "shape_rejected_low_fill",
                    label=int(i),
                    area=float(area[idx]),
                    fill=float(area[idx]) / rect_area,
                    required=float(TextRegions.pArea * 0.85),
                )
            elif aspect >= 18.0:
                _record("shape_rejected_bad_aspect", label=int(i), area=float(area[idx]), aspect=aspect)
            else:
                _record("shape_rejected_geometry", label=int(i), area=float(area[idx]), h=float(h), w=float(w))
            filt.append(f)
            R.append(rot)

        # filter bad regions:
        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        aidx = np.argsort(-area)

        good_sorted = good[filt][aidx]
        R_sorted = [R[i] for i in aidx]
        area_sorted = area[aidx]

        maxN = getattr(TextRegions, "maxRegionsForPlaneFit", None)
        if maxN is not None and maxN > 0 and len(good_sorted) > maxN:
            _record("shape_truncated_by_max_regions", kept=int(maxN), dropped=int(len(good_sorted) - maxN))
            good_sorted = good_sorted[:maxN]
            R_sorted = R_sorted[:maxN]
            area_sorted = area_sorted[:maxN]

        filter_info = {'label': good_sorted, 'rot': R_sorted, 'area': area_sorted}
        return filter_info

    @staticmethod
    def filter_depth(xyz, seg, regions, max_planes=6):
        """
        Быстрый отбор планарных регионов.

        + SKY-LIKE фильтр: большой верхний широкий регион пропускаем (обычно "небо")
        Делается ДЁШЕВО по flat_idx (bbox/area), до семплинга/плоскостей.

        Возвращает словарь:
        'label', 'area', 'coeff', 'inliers', 'rot'
        """
        import numpy as np
        from . import synth_utils as su

        xyz = np.asarray(xyz, dtype=np.float32)
        seg = np.asarray(seg, dtype=np.int32)

        labels = np.asarray(regions.get("label", []), dtype=np.int32)
        areas  = np.asarray(regions.get("area", []), dtype=np.float32)
        rots   = regions.get("rot", [None] * len(labels))

        plane_info = {
            'label': [],
            'coeff': [],
            'inliers': [],
            'area': [],
            'rot': [],
        }

        if labels.size == 0:
            return plane_info

        # сортировка по площади (убывание)
        order = np.argsort(-areas)
        labels = labels[order]
        areas  = areas[order]
        rots   = [rots[i] for i in order]

        H, W = seg.shape[:2]
        total_px = float(H * W)

        seg_flat = seg.reshape(-1)
        xyz_flat = xyz.reshape(-1, 3)

        # настройки семплинга (можно переопределить снаружи)
        max_points = int(getattr(TextRegions, "max_points_for_plane", 15000))
        min_points = int(getattr(TextRegions, "min_points_for_plane", 2500))

        # RANSAC
        trials = int(getattr(TextRegions, "ransac_fit_trials", 80))
        nsample = int(getattr(TextRegions, "ransac_nsample", 20))
        dist_thresh = float(getattr(TextRegions, "dist_thresh", 0.30))
        min_z_proj  = float(getattr(TextRegions, "min_z_projection", 0.05))

        # инлаеры
        inlier_ratio = float(getattr(TextRegions, "inlier_ratio", 0.10))
        min_inlier_abs = int(getattr(TextRegions, "min_inlier_abs", 60))

        verbose = bool(getattr(TextRegions, "verbose", False))
        ransac_debug = bool(getattr(TextRegions, "ransac_debug", False))
        def _log(msg):
            if verbose or ransac_debug:
                print(msg)

        stats_collector = getattr(TextRegions, "stats_collector", None)

        def _record(event, **fields):
            if stats_collector is None:
                return
            try:
                item = {"event": event}
                item.update(fields)
                stats_collector.append(item)
            except Exception:
                pass

        # --- SKY-like thresholds (tunable via class attrs) ---
        skip_sky = bool(getattr(TextRegions, "skip_sky_like", True))
        sky_min_area_frac = float(getattr(TextRegions, "sky_min_area_frac", 0.18))
        sky_max_y_center  = float(getattr(TextRegions, "sky_max_y_center", 0.38))
        sky_min_w_frac    = float(getattr(TextRegions, "sky_min_w_frac", 0.45))
        sky_max_h_frac    = float(getattr(TextRegions, "sky_max_h_frac", 0.75))

        def is_sky_like_from_indices(flat_idx, area_px):
            """
            Эвристика: большой верхний широкий регион похож на небо.
            flat_idx: индексы пикселей (в seg_flat)
            area_px: площадь региона в пикселях (можно из regions['area'])
            """
            if flat_idx.size == 0:
                return False

            area_frac = float(area_px) / max(total_px, 1.0)

            ys = (flat_idx // W).astype(np.int32, copy=False)
            xs = (flat_idx - ys * W).astype(np.int32, copy=False)

            y_min = int(ys.min()); y_max = int(ys.max())
            x_min = int(xs.min()); x_max = int(xs.max())

            h_frac = float(y_max - y_min + 1) / max(float(H), 1.0)
            w_frac = float(x_max - x_min + 1) / max(float(W), 1.0)
            y_center_frac = (0.5 * (y_min + y_max)) / max(float(H), 1.0)

            cond_area = area_frac >= sky_min_area_frac
            cond_top  = y_center_frac <= sky_max_y_center
            cond_wide = w_frac >= sky_min_w_frac
            cond_not_too_tall = h_frac <= sky_max_h_frac

            return bool(cond_area and cond_top and cond_wide and cond_not_too_tall)

        max_trials = getattr(TextRegions, "maxPlaneTrials", None)
        region_workers = int(getattr(TextRegions, "region_workers", 1) or 1)

        def _ls_fallback(pt, n_pt, lbl, min_inlier, log):
            try:
                X = np.c_[pt[:, 0], pt[:, 1], np.ones(n_pt, dtype=np.float32)]
                y = -pt[:, 2]
                coeff_ls, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                a_c, b_c, d_c = coeff_ls
                coeff = np.array([a_c, b_c, 1.0, d_c], dtype=np.float32)
                norm = float(np.linalg.norm(coeff[:3]))
                if not np.isfinite(norm) or norm <= 1e-8:
                    log(f"[filter_depth] LS fallback rejected for label={lbl}: bad coeff={coeff}")
                    _record("ls_rejected_bad_coeff", label=int(lbl), n_sample=int(n_pt))
                    return None
                coeff = coeff / norm
                dists = np.abs(pt.dot(coeff[:3]) + coeff[3])
                inliers = dists < dist_thresh
                nin = int(np.sum(inliers))
                if nin < int(min_inlier):
                    log(
                        f"[filter_depth] LS fallback rejected for label={lbl}: "
                        f"inliers={nin}/{n_pt} required={int(min_inlier)}"
                    )
                    _record(
                        "ls_rejected_low_inliers",
                        label=int(lbl),
                        inliers=nin,
                        required=int(min_inlier),
                        n_sample=int(n_pt),
                    )
                    return None
                log(f"[filter_depth] RANSAC failed for label={lbl}, LS fallback accepted with inliers={nin}/{n_pt}")
                _record("ls_accepted", label=int(lbl), inliers=nin, required=int(min_inlier), n_sample=int(n_pt))
                return coeff.astype(np.float32), inliers
            except Exception as e:
                log(f"[filter_depth] LS fallback failed for label={lbl}: {repr(e)}")
                _record("ls_exception", label=int(lbl), error=repr(e))
                return None

        def _fit_candidate(idx, lbl, a, r, rng=None):
            logs = []
            lbl = int(lbl)

            def log(msg):
                if verbose or ransac_debug:
                    logs.append(msg)

            flat_idx = np.flatnonzero(seg_flat == lbl)
            n_full = int(flat_idx.size)

            if n_full < min_points:
                _record("too_few_region_points", label=int(lbl), n_full=n_full, min_points=int(min_points))
                return idx, None, logs

            if skip_sky:
                try:
                    area_px = float(a) if np.isfinite(a) and a > 0 else float(n_full)
                    if is_sky_like_from_indices(flat_idx, area_px):
                        area_frac = float(area_px) / max(total_px, 1.0)
                        ys = (flat_idx // W).astype(np.int32, copy=False)
                        xs = (flat_idx - ys * W).astype(np.int32, copy=False)
                        y_center_frac = (0.5 * (int(ys.min()) + int(ys.max()))) / max(float(H), 1.0)
                        w_frac = float(int(xs.max()) - int(xs.min()) + 1) / max(float(W), 1.0)
                        h_frac = float(int(ys.max()) - int(ys.min()) + 1) / max(float(H), 1.0)

                        log(
                            f"[filter_depth] skip label={lbl} as sky-like: "
                            f"area_frac={area_frac:.3f}, y_center={y_center_frac:.3f}, "
                            f"w_frac={w_frac:.3f}, h_frac={h_frac:.3f}"
                        )
                        _record(
                            "sky_like",
                            label=int(lbl),
                            area_frac=area_frac,
                            y_center=y_center_frac,
                            w_frac=w_frac,
                            h_frac=h_frac,
                        )
                        return idx, None, logs
                except Exception:
                    pass

            if rng is None:
                rng = np.random

            if n_full > max_points:
                if hasattr(rng, "choice"):
                    choose = rng.choice(n_full, max_points, replace=False)
                else:
                    choose = np.random.choice(n_full, max_points, replace=False)
                samp_idx = flat_idx[choose]
            else:
                samp_idx = flat_idx

            pt = xyz_flat[samp_idx].astype(np.float32, copy=False)
            n_pt = int(pt.shape[0])

            if n_pt < 200:
                _record("too_few_sample_points", label=int(lbl), n_sample=n_pt)
                return idx, None, logs

            if hasattr(rng, "integers"):
                nn_idx = rng.integers(0, n_pt, size=(nsample, trials), dtype=np.int32)
            else:
                nn_idx = np.random.randint(0, n_pt, size=(nsample, trials), dtype=np.int32)

            min_inlier = int(max(min_inlier_abs, int(inlier_ratio * n_pt)))
            min_inlier = int(min(min_inlier, n_pt))

            plane_model = su.isplanar(
                pt,
                nn_idx,
                dist_thresh,
                min_inlier,
                min_z_proj,
                debug=ransac_debug,
                debug_prefix=f"[RANSAC label={lbl}]",
                nsample=nsample,
            )

            if plane_model is None:
                _record("ransac_failed", label=int(lbl), n_sample=int(n_pt), required=int(min_inlier))
                plane_model = _ls_fallback(pt, n_pt, lbl, min_inlier, log)
                if plane_model is None:
                    return idx, None, logs
            else:
                try:
                    inl = plane_model[1]
                    nin = int(np.asarray(inl).sum()) if np.asarray(inl).dtype == np.bool_ else len(inl)
                except Exception:
                    nin = -1
                _record("ransac_accepted", label=int(lbl), inliers=nin, required=int(min_inlier), n_sample=int(n_pt))

            coeff, inliers = plane_model

            if abs(float(coeff[2])) <= (min_z_proj * 0.5):
                log(f"[filter_depth] label={lbl} weak z-normal: coeff={coeff}")
                _record("weak_z_normal", label=int(lbl), coeff_z=float(coeff[2]))

            result = {
                "label": lbl,
                "coeff": np.asarray(coeff, dtype=np.float32),
                "inliers": inliers,
                "area": float(a),
                "rot": r,
            }
            log(
                f"[filter_depth] accepted label={lbl}, "
                f"area={float(a):.1f}, n_full={n_full}, n_sample={n_pt}, "
                f"min_inlier={min_inlier}"
            )
            _record("depth_region_accepted", label=int(lbl), area=float(a), n_full=n_full, n_sample=n_pt)
            return idx, result, logs

        if region_workers > 1 and labels.size > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            candidates = []
            for idx, (lbl, a, r) in enumerate(zip(labels, areas, rots)):
                if (max_trials is not None) and (idx >= int(max_trials)):
                    _log(f"[filter_depth] reached maxPlaneTrials={max_trials}, stop scanning regions")
                    break
                candidates.append((idx, int(lbl), float(a), r))

            if candidates:
                max_workers = max(1, min(int(region_workers), len(candidates)))
                seed0 = int(np.random.randint(0, 2**31 - 1))
                results = []
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = []
                    for idx, lbl, a, r in candidates:
                        rng = np.random.default_rng(seed0 + int(idx))
                        futures.append(pool.submit(_fit_candidate, idx, lbl, a, r, rng))
                    for fut in as_completed(futures):
                        results.append(fut.result())

                for _idx, result, logs in sorted(results, key=lambda item: item[0]):
                    for msg in logs:
                        _log(msg)
                    if result is None:
                        continue
                    if len(plane_info['label']) >= int(max_planes):
                        break
                    plane_info['label'].append(result["label"])
                    plane_info['coeff'].append(result["coeff"])
                    plane_info['inliers'].append(result["inliers"])
                    plane_info['area'].append(result["area"])
                    plane_info['rot'].append(result["rot"])

                if plane_info['coeff']:
                    plane_info['label'] = np.asarray(plane_info['label'], dtype=np.int32)
                    plane_info['area']  = np.asarray(plane_info['area'], dtype=np.float32)
                    plane_info['coeff'] = np.asarray(plane_info['coeff'], dtype=np.float32)
                else:
                    plane_info['label'] = np.zeros((0,), dtype=np.int32)
                    plane_info['area']  = np.zeros((0,), dtype=np.float32)
                    plane_info['coeff'] = np.zeros((0, 4), dtype=np.float32)
                    plane_info['rot']   = []

                return plane_info

        for idx, (lbl, a, r) in enumerate(zip(labels, areas, rots)):
            if len(plane_info['label']) >= int(max_planes):
                break

            if (max_trials is not None) and (idx >= int(max_trials)):
                _log(f"[filter_depth] reached maxPlaneTrials={max_trials}, stop scanning regions")
                break

            lbl = int(lbl)

            # flat индексы пикселей данного сегмента
            flat_idx = np.flatnonzero(seg_flat == lbl)
            n_full = int(flat_idx.size)

            if n_full < min_points:
                _record("too_few_region_points", label=int(lbl), n_full=n_full, min_points=int(min_points))
                continue

            # --- SKY-LIKE FILTER (до семплинга/плоскостей) ---
            if skip_sky:
                try:
                    # a уже "area" из regions; обычно это число пикселей сегмента
                    area_px = float(a) if np.isfinite(a) and a > 0 else float(n_full)
                    if is_sky_like_from_indices(flat_idx, area_px):
                        area_frac = float(area_px) / max(total_px, 1.0)
                        ys = (flat_idx // W).astype(np.int32, copy=False)
                        xs = (flat_idx - ys * W).astype(np.int32, copy=False)
                        y_center_frac = (0.5 * (int(ys.min()) + int(ys.max()))) / max(float(H), 1.0)
                        w_frac = float(int(xs.max()) - int(xs.min()) + 1) / max(float(W), 1.0)
                        h_frac = float(int(ys.max()) - int(ys.min()) + 1) / max(float(H), 1.0)

                        _log(
                            f"[filter_depth] skip label={lbl} as sky-like: "
                            f"area_frac={area_frac:.3f}, y_center={y_center_frac:.3f}, "
                            f"w_frac={w_frac:.3f}, h_frac={h_frac:.3f}"
                        )
                        _record(
                            "sky_like",
                            label=int(lbl),
                            area_frac=area_frac,
                            y_center=y_center_frac,
                            w_frac=w_frac,
                            h_frac=h_frac,
                        )
                        continue
                except Exception:
                    # если что-то пошло не так — не ломаем пайплайн
                    pass

            # СЭМПЛИРУЕМ точки региона
            if n_full > max_points:
                choose = np.random.choice(n_full, max_points, replace=False)
                samp_idx = flat_idx[choose]
            else:
                samp_idx = flat_idx

            pt = xyz_flat[samp_idx].astype(np.float32, copy=False)
            n_pt = int(pt.shape[0])

            if n_pt < 200:
                _record("too_few_sample_points", label=int(lbl), n_sample=n_pt)
                continue

            # nn_idx — быстрый вариант (рандом)
            nn_idx = np.random.randint(0, n_pt, size=(nsample, trials), dtype=np.int32)

            min_inlier = int(max(min_inlier_abs, int(inlier_ratio * n_pt)))
            min_inlier = int(min(min_inlier, n_pt))

            plane_model = su.isplanar(
                pt,
                nn_idx,
                dist_thresh,
                min_inlier,
                min_z_proj,
                debug=ransac_debug,
                debug_prefix=f"[RANSAC label={lbl}]",
                nsample=nsample,
            )

            if plane_model is None:
                _record("ransac_failed", label=int(lbl), n_sample=int(n_pt), required=int(min_inlier))
                plane_model = _ls_fallback(pt, n_pt, lbl, min_inlier, _log)
                if plane_model is None:
                    continue
            else:
                try:
                    inl = plane_model[1]
                    nin = int(np.asarray(inl).sum()) if np.asarray(inl).dtype == np.bool_ else len(inl)
                except Exception:
                    nin = -1
                _record("ransac_accepted", label=int(lbl), inliers=nin, required=int(min_inlier), n_sample=int(n_pt))

            coeff, inliers = plane_model

            # мягкая проверка по нормали
            if abs(float(coeff[2])) <= (min_z_proj * 0.5):
                _log(f"[filter_depth] label={lbl} weak z-normal: coeff={coeff}")
                _record("weak_z_normal", label=int(lbl), coeff_z=float(coeff[2]))

            plane_info['label'].append(lbl)
            plane_info['coeff'].append(np.asarray(coeff, dtype=np.float32))
            plane_info['inliers'].append(inliers)
            plane_info['area'].append(float(a))
            plane_info['rot'].append(r)

            _log(
                f"[filter_depth] accepted label={lbl}, "
                f"area={float(a):.1f}, n_full={n_full}, n_sample={n_pt}, "
                f"min_inlier={min_inlier}, kept={len(plane_info['label'])}"
            )
            _record("depth_region_accepted", label=int(lbl), area=float(a), n_full=n_full, n_sample=n_pt)

        # привести к массивам
        if plane_info['coeff']:
            plane_info['label'] = np.asarray(plane_info['label'], dtype=np.int32)
            plane_info['area']  = np.asarray(plane_info['area'], dtype=np.float32)
            plane_info['coeff'] = np.asarray(plane_info['coeff'], dtype=np.float32)
        else:
            plane_info['label'] = np.zeros((0,), dtype=np.int32)
            plane_info['area']  = np.zeros((0,), dtype=np.float32)
            plane_info['coeff'] = np.zeros((0, 4), dtype=np.float32)
            plane_info['rot']   = []

        return plane_info



    @staticmethod
    def get_regions(xyz, seg, area, label):
        """
        Берём регионы напрямую из сегментации `seg`, полностью
        игнорируя старые вектора `area` и `label` из h5.

        Это:
        - увеличивает количество кандидатов (масок),
        - гарантирует, что работаем именно с текущей сегментацией.
        """
        import numpy as np

        seg_np = np.asarray(seg)
        labels, counts = np.unique(seg_np, return_counts=True)

        # 0 считаем фоном — выкидываем
        keep = (labels != 0)
        labels = labels[keep]
        areas = counts[keep].astype(np.float32)

        # дальше используем уже существующий фильтр по форме
        return TextRegions.filter(seg_np, areas, labels)
def get_text_placement_mask(xyz, mask, plane, pad=2, viz=False,
                            persp_strength=1.0, max_tilt_deg=None):
    import scipy.spatial.distance as ssd
    import matplotlib.pyplot as plt
    from . import synth_utils as su

    contour, hier = cv2.findContours(mask.copy().astype('uint8'),
                                     mode=cv2.RETR_CCOMP,
                                     method=cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contour = [np.squeeze(c).astype('float') for c in contour]
    H, W = mask.shape[:2]

    # bring the contour 3d points to fronto-parallel config:
    pts, pts_fp = [], []
    center = np.array([W, H]) / 2.0
    n_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    for i in range(len(contour)):
        cnt_ij = contour[i]

        # 3D точки контура на плоскости
        xyz_ij = su.DepthCamera.plane2xyz(center, cnt_ij, plane)

        # БЫЛО: R = su.rot3d(plane[:3], n_front)
        # СТАЛО: ослабляем "силу" rectification, чтобы меньше давило перспективой
        R = rot3d_scaled(plane[:3], n_front,
                         strength=float(persp_strength),
                         max_tilt_deg=float(max_tilt_deg) if max_tilt_deg is not None else None)

        xyz_ij = xyz_ij.dot(R.T)
        pts_fp.append(xyz_ij[:, :2])
        pts.append(cnt_ij)

    # unrotate in 2D plane:
    rect = cv2.minAreaRect(pts_fp[0].copy().astype('float32'))
    box = np.array(cv2.boxPoints(rect))
    R2d = su.unrotate2d(box.copy())
    box = np.vstack([box, box[0, :]])  # close for viz

    mu = np.median(pts_fp[0], axis=0)
    pts_tmp = (pts_fp[0] - mu[None, :]).dot(R2d.T) + mu[None, :]
    boxR = (box - mu[None, :]).dot(R2d.T) + mu[None, :]

    # rescale to approx target region:
    s = rescale_frontoparallel(pts_tmp, boxR, pts[0])
    boxR *= s
    for i in range(len(pts_fp)):
        pts_fp[i] = s * ((pts_fp[i] - mu[None, :]).dot(R2d.T) + mu[None, :])

    # paint the unrotated contour points:
    minxy = -np.min(boxR, axis=0)
    ROW = np.max(ssd.pdist(np.atleast_2d(boxR[:, 0]).T))
    COL = np.max(ssd.pdist(np.atleast_2d(boxR[:, 1]).T))

    # (твои “полотна/поля” оставляю как было)
    ROW *= 1.12
    COL *= 1.06
    pad = max(int(pad), 14)

    place_mask = 255 * np.ones((int(np.ceil(COL)) + pad, int(np.ceil(ROW)) + pad), 'uint8')

    pts_fp_i32 = [(pts_fp[i] + (minxy + pad // 2)[None, :]).astype('int32') for i in range(len(pts_fp))]
    cv2.drawContours(place_mask, pts_fp_i32, -1, 0,
                     thickness=cv2.FILLED, lineType=8, hierarchy=hier)

    if not TextRegions.filter_rectified((~place_mask).astype('float') / 255):
        return

    # calculate the homography
    Hm, _ = cv2.findHomography(pts[0].astype('float32').copy(),
                              pts_fp_i32[0].astype('float32').copy(), method=0)
    Hinv, _ = cv2.findHomography(pts_fp_i32[0].astype('float32').copy(),
                                 pts[0].astype('float32').copy(), method=0)

    if viz:
        plt.subplot(1, 2, 1); plt.imshow(mask)
        plt.subplot(1, 2, 2); plt.imshow(~place_mask)
        for i in range(len(pts_fp_i32)):
            plt.scatter(pts_fp_i32[i][:, 0], pts_fp_i32[i][:, 1],
                        edgecolors='none', facecolor='g', alpha=0.5)
        plt.show()

    return place_mask, Hm, Hinv
