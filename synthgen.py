# Author: Ankush Gupta
# Date: 2015

"""
Main script for synthetic text rendering.
"""

from __future__ import division
import copy
import cv2 as cv
import h5py
from PIL import Image
#import mayavi.mlab as mym
import matplotlib.pyplot as plt
import random
plt.ion()
try:
    import matplotlib
    print("[VIZ] matplotlib backend:", matplotlib.get_backend())
    # «Прогрев» окна — снижает шанс, что первое изображение «проглотится»
    _ = plt.figure(99); plt.plot([0,1],[0,1]); plt.show(block=False); plt.pause(0.05); plt.close(99)
except Exception:
    pass

from noise_utils import apply_random_augmentations
import os.path as osp
import scipy.ndimage as sim
import scipy.spatial.distance as ssd
import synth_utils as su
import text_utils as tu
from colorize3_poisson import Colorize
from common import *
import traceback, itertools
import numpy as np

# --- OpenCV aliases (поддерживаем и cv2, и cv) ---
try:
    import cv2
except Exception as e:
    raise

# если алиас cv не задан, делаем его ссылкой на cv2
try:
    cv
except NameError:
    cv = cv2

MIN_FONT_PX   = 14      # минимально допустимая высота шрифта (под себя)
SHRINK_STEP   = 0.90    # шаг уменьшения шрифта при бэкоффе
SIDE_MARGIN   = 0.8    # внутренний отступ по краям rectified-региона
SPACE_FACTOR  = 0.3
MIN_RECTIFIED_MASK_SUM = 1200

# Совместимость со старым кодом на NumPy 2.x
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

DEBUG = True

import os

def _warp_points(Hinv, pts_xy):
    """pts_xy: (N,2) в FP -> (N,2) в IMG"""
    import numpy as np
    pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    pts = np.concatenate([pts_xy, np.ones((len(pts_xy), 1), np.float32)], axis=1).T  # 3xN
    w = np.asarray(Hinv, dtype=np.float32) @ pts
    w /= (w[2:3, :] + 1e-6)
    return w[:2, :].T

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
    min_z_projection = 0.05   # было 0.15 — почти любую «фронтальную» нормаль пускаем

    minW = 16

    # <<< НОВОЕ: ограничиваем число регионов, куда полезем с RANSAC и плоскостями >>>
    maxRegionsForPlaneFit = 15  # максимум регионов после TextRegions.filter
    maxPlaneTrials = 15         # максимум регионов в TextRegions.filter_depth

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
        good = label[area > TextRegions.minArea]
        area = area[area > TextRegions.minArea]
        filt, R = [], []
        for idx, i in enumerate(good):
            mask = (seg == i)

            # np.where -> (rows, cols) = (y, x)
            ys, xs = np.where(mask)

            # OpenCV ждёт точки (x, y)
            coords = np.c_[xs, ys].astype('float32')

            if coords.shape[0] < 10:
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
        import synth_utils as su

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
        dist_thresh = float(getattr(TextRegions, "dist_thresh", 0.30))
        min_z_proj  = float(getattr(TextRegions, "min_z_projection", 0.05))

        # инлаеры
        inlier_ratio = float(getattr(TextRegions, "inlier_ratio", 0.10))
        min_inlier_abs = int(getattr(TextRegions, "min_inlier_abs", 60))

        verbose = bool(getattr(TextRegions, "verbose", False))
        def _log(msg):
            if verbose:
                print(msg)

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
                continue

            # nn_idx — быстрый вариант (рандом)
            nn_idx = np.random.randint(0, n_pt, size=(5, trials), dtype=np.int32)

            min_inlier = int(max(min_inlier_abs, int(inlier_ratio * n_pt)))
            min_inlier = int(min(min_inlier, n_pt))

            plane_model = su.isplanar(
                pt,
                nn_idx,
                dist_thresh,
                min_inlier,
                min_z_proj
            )

            if plane_model is None:
                # LS fallback тоже по pt (sampled)
                try:
                    X = np.c_[pt[:, 0], pt[:, 1], np.ones(n_pt, dtype=np.float32)]
                    y = -pt[:, 2]
                    coeff_ls, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    a_c, b_c, d_c = coeff_ls
                    coeff = np.array([a_c, b_c, 1.0, d_c], dtype=np.float32)
                    inliers = np.arange(n_pt, dtype=np.int32)
                    plane_model = (coeff, inliers)
                    _log(f"[filter_depth] RANSAC failed for label={lbl}, LS fallback accepted")
                except Exception as e:
                    _log(f"[filter_depth] LS fallback failed for label={lbl}: {repr(e)}")
                    continue

            coeff, inliers = plane_model

            # мягкая проверка по нормали
            if abs(float(coeff[2])) <= (min_z_proj * 0.5):
                _log(f"[filter_depth] label={lbl} weak z-normal: coeff={coeff}")

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
    
def estimate_local_scale_grid(Hinv, free_mask_fp, k=9, delta=6, seed=None):
    """
    Оценка локального масштаба FP->IMG для homography Hinv.

    Идея: берём несколько точек в свободной области (free_mask_fp==1),
    и считаем |dX/dx| и |dY/dy| через finite-diff в image-space.

    Возвращает скаляр scale ~ px_img / px_fp (median по точкам).
    """

    if Hinv is None or free_mask_fp is None:
        return 1.0

    Hinv = np.asarray(Hinv, dtype=np.float64)
    if Hinv.shape != (3, 3) or (not np.isfinite(Hinv).all()):
        return 1.0

    m = np.asarray(free_mask_fp).astype(np.uint8)
    if m.ndim != 2:
        return 1.0

    H_fp, W_fp = m.shape[:2]
    d = int(max(1, delta))

    # чтобы (x+d, y) и (x, y+d) не вылезали за границу
    ys, xs = np.where(m > 0)
    if xs.size < 16:
        return 1.0

    ok = (xs >= d) & (xs < (W_fp - d)) & (ys >= d) & (ys < (H_fp - d))
    xs = xs[ok]; ys = ys[ok]
    if xs.size < 8:
        return 1.0

    if seed is not None:
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(xs.size, size=min(int(k), xs.size), replace=False)
    else:
        pick = np.random.choice(xs.size, size=min(int(k), xs.size), replace=False)

    x0 = xs[pick].astype(np.float32)
    y0 = ys[pick].astype(np.float32)

    P  = np.stack([x0,       y0      ], axis=1)
    Px = np.stack([x0 + d,   y0      ], axis=1)
    Py = np.stack([x0,       y0 + d  ], axis=1)

    def warp(pts):
        pts_cv = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts_cv, Hinv).reshape(-1, 2)
        return out

    try:
        W0 = warp(P)
        Wx = warp(Px)
        Wy = warp(Py)
    except Exception:
        return 1.0

    dx = np.linalg.norm(Wx - W0, axis=1) / float(d)
    dy = np.linalg.norm(Wy - W0, axis=1) / float(d)

    s = 0.5 * (dx + dy)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 1.0

    out = float(np.median(s))
    if not np.isfinite(out) or out <= 1e-6:
        return 1.0
    return out


def rescale_frontoparallel(p_fp, box_fp, p_im):
    """
    The fronto-parallel image region is rescaled to bring it in 
    the same approx. size as the target region size.

    p_fp : nx2 coordinates of contour points in the fronto-parallel plane
    box  : 4x2 coordinates of bounding box of p_fp
    p_im : nx2 coordinates of contour in the image

    NOTE : p_fp and p are corresponding, i.e. : p_fp[i] ~ p[i]

    Returns the scale 's' to scale the fronto-parallel points by.
    """
    # Вычисляем длины сторон бокса
    l1 = np.linalg.norm(box_fp[1, :] - box_fp[0, :])
    l2 = np.linalg.norm(box_fp[1, :] - box_fp[2, :])
    
    # Находим индексы ближайших точек
    n0 = np.argmin(np.linalg.norm(p_fp - box_fp[0, :][None, :], axis=1))
    n1 = np.argmin(np.linalg.norm(p_fp - box_fp[1, :][None, :], axis=1))
    n2 = np.argmin(np.linalg.norm(p_fp - box_fp[2, :][None, :], axis=1))

    # Проверяем индексы на корректность
    if n0 < 0 or n1 < 0 or n2 < 0 or n0 >= len(p_im) or n1 >= len(p_im) or n2 >= len(p_im):
        return 1.0  # Возвращаем дефолтное значение для масштаба

    # Вычисляем расстояния между точками
    lt1 = np.linalg.norm(p_im[n1, :] - p_im[n0, :])
    lt2 = np.linalg.norm(p_im[n1, :] - p_im[n2, :])

    # Проверяем на наличие бесконечных значений или NaN
    if np.isinf(lt1) or np.isinf(lt2) or np.isnan(lt1) or np.isnan(lt2):
        return 1.0  # Возвращаем дефолтное значение для масштаба

    # Рассчитываем масштаб
    s = max(lt1 / l1, lt2 / l2)
    
    # Проверяем на бесконечность или NaN
    if not np.isfinite(s):
        s = 1.0  # Возвращаем дефолтное значение для масштаба

    return s

def _normalize(v, eps=1e-8):
    import numpy as np
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def rot3d_scaled(n_src, n_dst, strength=1.0, max_tilt_deg=None):
    """
    Поворот, который НЕ полностью выравнивает n_src->n_dst,
    а делает это с силой strength (0..1): angle_scaled = angle * strength.
    Можно ограничить максимальный tilt (max_tilt_deg), чтобы экстремальные плоскости не ломали геометрию.
    """
    import numpy as np

    strength = float(np.clip(strength, 0.0, 1.0))
    n0 = _normalize(n_src)
    n1 = _normalize(n_dst)

    c = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
    angle = float(np.arccos(c))  # 0..pi

    if max_tilt_deg is not None:
        max_tilt = float(np.deg2rad(max_tilt_deg))
        angle = min(angle, max_tilt)

    angle *= strength

    axis = np.cross(n0, n1)
    s = float(np.linalg.norm(axis))
    if s < 1e-8 or angle < 1e-8:
        return np.eye(3, dtype=np.float32)

    axis = axis / s
    x, y, z = axis.astype(np.float32)

    K = np.array([
        [0.0, -z,   y],
        [z,   0.0, -x],
        [-y,  x,   0.0],
    ], dtype=np.float32)

    I = np.eye(3, dtype=np.float32)
    sa = float(np.sin(angle))
    ca = float(np.cos(angle))

    R = I + sa * K + (1.0 - ca) * (K @ K)
    return R.astype(np.float32)



def get_text_placement_mask(xyz, mask, plane, pad=2, viz=False,
                            persp_strength=1.0, max_tilt_deg=45.0):
    import scipy.spatial.distance as ssd
    import matplotlib.pyplot as plt
    import synth_utils as su

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



def _rgb(im):
    import numpy as _np, cv2 as _cv2
    if im is None:
        return im
    if im.ndim == 3 and im.shape[2] == 3:
        # Предполагаем, что массив может быть BGR -> приводим к RGB для matplotlib
        return _cv2.cvtColor(im, _cv2.COLOR_BGR2RGB)
    return im

def to_rgb(arr):
    import numpy as _np, cv2 as _cv2
    if arr is None: return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        # считаем, что «внутренний стандарт» — RGB;
        # если массив пришёл из OpenCV/BGR, явно переведём.
        # Здесь безопасно всегда переворачивать каналы, если приходят из cv2.
        return _cv2.cvtColor(arr, _cv2.COLOR_BGR2RGB)
    return arr

def _plt_stable_draw(fig=None, pause=0.25):
    import matplotlib.pyplot as _plt
    try:
        f = fig or _plt.gcf()
        f.canvas.draw_idle()
        f.canvas.flush_events()
        _plt.show(block=False)   # держим окно живым
        _plt.pause(pause)
        return True
    except Exception as e:
        print("[VIZ] Matplotlib draw failed:", e)
        return False

def _cv2_preview(win_name, img_rgb):
    if img_rgb is None:
        return
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_rgb
    cv2.imshow(win_name, bgr)
    cv2.waitKey(1)

def viz_textbb(fignum,text_im, bb_list,alpha=1.0):
    import matplotlib.pyplot as plt
    plt.figure(fignum)
    plt.clf()
    plt.imshow(_rgb(text_im))
    H,W = text_im.shape[:2]
    for i in range(len(bb_list)):
        bbs = bb_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            plt.plot(bb[0,:], bb[1,:], 'r', linewidth=2, alpha=alpha)
    plt.gca().set_xlim([0,W-1]); plt.gca().set_ylim([H-1,0])
    plt.tight_layout()
    if not _plt_stable_draw(plt.gcf(), pause=0.35):
        _cv2_preview("SynthText (bb)", text_im)

def viz_masks(fignum,rgb,seg,depth,label):
    import matplotlib.pyplot as plt, numpy as np, cv2
    def mean_seg(rgb,seg,label):
        mim = np.zeros_like(rgb)
        for i in np.unique(seg.flat):
            mask = seg==i
            col = np.mean(rgb[mask,:],axis=0)
            mim[mask,:] = col[None,None,:]
        mim[seg==0,:] = 0
        return mim

    mim = mean_seg(rgb,seg,label)
    img = rgb.copy()
    for i,idx in enumerate(label):
        mask = seg==idx
        rgb_rand = (255*np.random.rand(3)).astype('uint8')
        img[mask] = rgb_rand[None,None,:]

    plt.figure(fignum)
    plt.clf()
    ims = [rgb,mim,depth,img]
    for i in range(len(ims)):
        plt.subplot(2,2,i+1)
        plt.imshow(_rgb(ims[i]))
    plt.tight_layout()
    if not _plt_stable_draw(plt.gcf(), pause=0.35):
        _cv2_preview("SynthText (masks)", rgb)

class   RendererV3(object):

    def __init__(self, data_dir, max_time=None):
        self.text_renderer = tu.RenderFont(data_dir)
        self.colorizer = Colorize(data_dir)

        self.max_time = max_time

        from collections import deque
        self._word_queue = deque()

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

        # --- filter_for_placement: параметры rectification (нужны, иначе дефолты) ---
        self.persp_strength = 0.6
        self.persp_max_tilt_deg = 45.0

        # --- homography / FP debug (используются) ---
        self.debug_hgeom = True
        self.debug_hgeom_max_regions = 3
        self.debug_hgeom_npts = 64
        self.debug_hgeom_print_mats = False

        # --- overlay perspective boost (используются) ---
        self.overlay_min_persp_strength = 0.1
        self.overlay_persp_boost = 4
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

        self.debug = True

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

        self.disable_all_augs = False
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

        # параметры ослабления перспективы (можешь задать в __init__)
        persp_strength = float(getattr(self, "persp_strength", 0.6))
        persp_max_tilt_deg = getattr(self, "persp_max_tilt_deg", 45.0)

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
        Быстрый выбор региона через кэш.

        НОВОЕ:
        - force_ireg: если задан, пробуем ТОЛЬКО этот регион (для "перемешали регионы и идём по списку")
        """
        import numpy as np

        if "min_char_px_img" in kwargs:
            try:
                min_text_px_img = int(kwargs.get("min_char_px_img"))
            except Exception:
                pass

        if img is None:
            return None, None, None
        H_img, W_img = img.shape[:2]
        self._img_shape_last = (H_img, W_img)

        if (not hasattr(self, "_region_cache")) or (self._region_cache is None) or (not self._region_cache.get("candidates")):
            self._ensure_region_cache(img, place_masks, regions)

        cache = self._region_cache
        if not cache or not cache.get("candidates"):
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
                return occ < float(getattr(self, "occupied_bbox_max_frac", 0.15))
            except Exception:
                return True

        def _try_region(i: int):
            if i is None:
                return None, None, None
            i = int(i)

            if i < 0 or i >= len(cache["banned"]):
                return None, None, None
            if bool(cache["banned"][i]):
                return None, None, None
            if avoid_repeat and (i in used):
                return None, None, None
            if not _occupied_ok(cache["bbox_img"][i]):
                return None, None, None

            fp_w, fp_h = cache["fp_wh"][i]
            fp_w = float(fp_w); fp_h = float(fp_h)
            if fp_w <= 0 or fp_h <= 0:
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
                return None, None, None

            f_nom = f_max * float(fill)
            f_min_req = float(min_text_px_img) / max(1e-6, s)
            f_fit = float(max(f_nom, f_min_req, float(min_font_px)))
            if f_fit > f_max:
                return None, None, None

            base_ang = float(cache["base_angle"][i])
            selected_angle = base_ang + float(np.random.uniform(-angle_jitter, angle_jitter))

            if avoid_repeat:
                used.add(i)

            return i, f_fit, selected_angle

        # --- режим "принудительно этот регион" ---
        if force_ireg is not None:
            return _try_region(int(force_ireg))

        # --- стандартный режим (как было) ---
        cand = cache["candidates"]
        pick_pool = cand[:max(1, min(topk, len(cand)))]

        scores = np.array([cache["score"][i] for i in pick_pool], dtype=np.float64)
        scores = np.maximum(scores, 1e-9)
        probs = scores / scores.sum()

        for _ in range(int(tries)):
            i = int(np.random.choice(pick_pool, p=probs))
            out = _try_region(i)
            if out[0] is not None:
                return out

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

        H_img, W_img = img.shape[:2]
        self._img_shape_last = (H_img, W_img)

        if not place_masks:
            return None

        # --- font init ---
        try:
            fs = self.text_renderer.font_state.sample()
            font = self.text_renderer.font_state.init_font(fs)
            f_asp = float(self.text_renderer.font_state.get_aspect_ratio(font))
            if not np.isfinite(f_asp) or f_asp <= 1e-6:
                f_asp = 1.0
        except Exception:
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
        except Exception:
            return None

        # --- layout ---
        try:
            nline_raw, nchar_raw = self.text_renderer.get_nline_nchar((128, 512), f_layout, f_layout * f_asp)
        except Exception:
            nline_raw, nchar_raw = 1, 12

        nline_eff = max(1, int(nline_raw or 1))
        nchar_eff = 10 if (nchar_raw is None or int(nchar_raw) < 6) else int(nchar_raw)

        # --- text sample ---
        try:
            txt_str = self._sample_layout_text(nline_eff, nchar_eff, max_retries=5)
        except Exception:
            return None

        if not txt_str or not isinstance(txt_str, str) or not txt_str.strip():
            return None
        txt_str = txt_str.strip()

        # --- убедимся, что кэш регионов посчитан ---
        if (not hasattr(self, "_region_cache")) or (self._region_cache is None) or (not self._region_cache.get("candidates")):
            self._ensure_region_cache(img, place_masks, regions)
        cache = self._region_cache
        if not cache or not cache.get("candidates"):
            return None

        # --- select region (или force_ireg) ---
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
            return None

        # --- region quad + bbox + s_loc из кэша ---
        region_coords = cache["quad_img"][ireg]
        if region_coords is None:
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

        # --- проверим влезание в FP bbox ---
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
            nchar=nchar_eff,
            s_loc=round(float(s_loc), 3),
            near_w=round(float(near_w), 1),
            near_h=round(float(near_h), 1),
            fill=round(float(fill), 2),
            char_h_px=round(float(char_h_px), 1),
            f_fit=round(float(f_fit), 1),
            f_target_fp=round(float(f_target_fp), 1),
            f_final=round(float(f_final), 1),
            force_ireg=force_ireg
        )

        # failed cache
        key = (int(ireg), int(round(float(f_final))))
        if hasattr(self, "_failed_pairs") and (key in self._failed_pairs):
            return None

        # set font.size
        try:
            font.size = self.text_renderer.font_state.get_font_size(font, float(f_final))
        except Exception:
            return None

        # render overlay
        try:
            img_new, bb_img, text_mask_img = self.render_text_overlay(
                img, txt_str, font,
                selected_angle=selected_angle,
                region_coords=region_coords,
                depth=depth
            )
        except Exception:
            try:
                if hasattr(self, "_failed_pairs"):
                    self._failed_pairs.add(key)
            except Exception:
                pass
            return None

        if img_new is None or text_mask_img is None:
            try:
                if hasattr(self, "_failed_pairs"):
                    self._failed_pairs.add(key)
            except Exception:
                pass
            return None

        if int((text_mask_img > 0).sum()) == 0:
            try:
                if hasattr(self, "_failed_pairs"):
                    self._failed_pairs.add(key)
            except Exception:
                pass
            return None

        self.last_font_h_px = float(f_final)
        return img_new, txt_str, bb_img, text_mask_img



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
        Берёт слова последовательно из очереди.
        Если очередь пуста — набивает её из text_source.sample(...).
        Фоллбэка "пример" больше нет: если уж совсем нечего — берём любое слово без min_len.
        """
        import re
        from collections import deque

        if not hasattr(self, "_word_queue") or self._word_queue is None:
            self._word_queue = deque()

        min_len = int(getattr(self, "min_word_len", 4))

        def tokenize(s: str):
            return re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", s)

        # 1) если есть запас — отдаём следующее
        while self._word_queue:
            w = self._word_queue.popleft()
            if len(w) >= min_len:
                return w

        # 2) иначе пытаемся набить очередь
        last_raw_text = ""
        for _ in range(max_retries):
            try:
                kind = tu.sample_weighted(self.text_renderer.p_text)
            except Exception:
                kind = None

            try:
                raw_obj = self.text_renderer.text_source.sample(nline, nchar, kind)
            except Exception:
                raw_obj = None

            if isinstance(raw_obj, list):
                last_raw_text = " ".join(str(x).strip() for x in raw_obj if str(x).strip())
            elif isinstance(raw_obj, str):
                last_raw_text = raw_obj.strip()
            else:
                last_raw_text = ""

            if not last_raw_text:
                continue

            words = tokenize(last_raw_text)
            if not words:
                continue

            self._word_queue = deque(words)

            while self._word_queue:
                w = self._word_queue.popleft()
                if len(w) >= min_len:
                    return w

        # 3) если совсем не получилось — берём любое слово без min_len
        if last_raw_text:
            words = tokenize(last_raw_text)
            if words:
                return words[0]

        return "text"

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
        log = bool(getattr(self, "overlay_occ_log", True))

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

            Wc, Hc = self._overlay_canvas_size_from_quad(dst_quad, min_size=min_canvas, max_size=max_canvas, scale=canvas_scale)

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
            rgb_loc, a_loc = self._overlay_fit_rgba_into_canvas(rgb_loc, a_loc, fill=fill, thr=8, allow_upscale=True, max_up=max_up)

            # warp to image
            warped_rgb, warped_a_full = self._overlay_warp_rgba_to_image(rgb_loc, a_loc, dst_quad, (H_img, W_img))
            if warped_rgb is None or warped_a_full is None:
                return None, None, None

            if warped_a_full.ndim == 3:
                warped_a_full = warped_a_full[:, :, 0]
            if warped_a_full.dtype != np.uint8:
                warped_a_full = np.clip(warped_a_full, 0, 255).astype(np.uint8)

            alpha_thr = int(getattr(self, "overlay_alpha_thr", 2))
            alpha_thr = max(1, alpha_thr)

            # FULL mask (для GT/charBB при желании)
            text_mask_full = ((warped_a_full > alpha_thr).astype(np.uint8) * 255)
            if int(text_mask_full.sum()) == 0:
                text_mask_full = ((warped_a_full > 1).astype(np.uint8) * 255)
                if int(text_mask_full.sum()) == 0:
                    if debug:
                        print("[OVERLAY] warped mask empty (full)")
                    return None, None, None

            # --- synthetic controlled occlusion ---
            warped_a_vis = warped_a_full
            occ_mask_u8 = None
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

            # --- background rectangle ---
            # Ключевой переключатель:
            # True  -> фон под текстом только там, где текст ВИДИМ (после окклюзии)
            # False -> фон под текстом по ПОЛНОЙ альфе (чтобы окклюзия не "выбивалась" цветом)
            use_visible_alpha_for_bg = bool(getattr(self, "overlay_bg_use_visible_alpha", False))
            bg_alpha = warped_a_vis if use_visible_alpha_for_bg else warped_a_full

            bg_thr = int(getattr(self, "overlay_bg_alpha_thr", max(10, alpha_thr * 4)))
            img_bg = self._overlay_apply_bg_rect_imgspace(img, bg_alpha, pad_px=16, alpha_thr=bg_thr)

            # blend text (используем альфу ПОСЛЕ окклюзии)
            img_new = self._overlay_alpha_blend(img_bg, warped_rgb, warped_a_vis)

            # --- заполнение зоны окклюдера ---
            # Обычно НЕ нужно (альфа уже убита), но полезно если ты хочешь "гарантировать"
            # совпадение цвета/освещения именно с фоном под текстом.
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

                # чем заполнять “перекрытие”
                # "img_bg" -> совпадает с фоном под текстом (обычно самое стабильное)
                # "img"    -> совпадает с исходной сценой (если ты НЕ хочешь, чтобы bg-rect “оставался”)
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

        try:
            depth = np.asarray(depth)
            rgb = np.asarray(rgb)
            seg = np.asarray(seg)
        except Exception:
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
            if debug_regions:
                print("[render_text] start:", rgb.shape, seg_i.shape, "ninstance=", ninstance)

            xyz = su.DepthCamera.depth2xyz(depth_f)

            regions = TextRegions.get_regions(xyz, seg_i, area, label)
            if debug_regions:
                print("[render_text] get_regions ->", len(regions.get("label", [])))

            regions = TextRegions.filter_depth(xyz, seg_i, regions)
            if debug_regions:
                print("[render_text] filter_depth ->", len(regions.get("label", [])))

            regions = self.filter_for_placement(xyz, seg_i, regions, viz=False)
            if regions is None or len(regions.get("place_mask", [])) == 0:
                return []

            nregions = len(regions["place_mask"])
            if nregions < 1:
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

        except Exception as e:
            if debug_regions:
                print("[render_text] region prep error:", repr(e))
            return []

        res = []

        for _ in range(int(ninstance)):
            img = rgb.copy()
            itext, ibb = [], []
            occupied_global = np.zeros(img.shape[:2], dtype=np.uint8)

            self._used_regions_this_image = set()

            nregions = len(regions["place_mask"])
            if nregions <= 0:
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

            for ireg in region_order:
                if placed_count >= target_blocks:
                    break
                if tries_left <= 0:
                    break

                for _t in range(per_region_cap):
                    if placed_count >= target_blocks:
                        break
                    if tries_left <= 0:
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
                        continue

                    img_new, text, bb, warped_mask = txt_render_res
                    if img_new is None or warped_mask is None:
                        continue

                    m_img = (warped_mask > 0).astype(np.uint8) * 255
                    if int(m_img.sum()) == 0:
                        continue

                    overlap = cv2.bitwise_and(occupied_global, m_img)
                    if int(overlap.sum()) > 0:
                        continue

                    m_inflated = cv2.dilate(m_img, ker, 1)
                    occupied_global = np.maximum(occupied_global, m_inflated)

                    img = img_new
                    itext.append(text)
                    ibb.append(bb)
                    placed_count += 1
                    break  # 1 текст на регион

            if placed_count == 0:
                continue

            idict = {'img': img, 'txt': itext, 'charBB': None, 'wordBB': None}

            bbs_valid = [b for b in ibb if b is not None and hasattr(b, "shape") and b.size > 0]
            if bbs_valid:
                try:
                    idict['charBB'] = np.concatenate(bbs_valid, axis=2)
                except Exception:
                    idict['charBB'] = None

            if idict['charBB'] is not None:
                try:
                    H, W = img.shape[:2]
                    idict['wordBB'] = self.char2wordBB(
                        idict['charBB'].copy(),
                        ' '.join(itext),
                        pad_px=4, pad_rel=0.05, clamp_shape=(H, W)
                    )
                except Exception:
                    idict['wordBB'] = None

            # ============================
            # ✅ ШУМЫ / ДЕГРАДАЦИЯ ПОСЛЕ ВСЕГО
            # ============================
            if not disable_augs:
                try:
                    from noise_utils import apply_noise_recipe

                    cfg = getattr(self, "noise_cfg", None)

                    # можно настраивать через self:
                    p_none = float(getattr(self, "noise_p_none", 0.12))         # шанс “вообще без ауг”
                    p_boost = float(getattr(self, "noise_p_boost", 1.0))        # чаще
                    strength = float(getattr(self, "noise_strength", 1.0))      # сильнее
                    force_one = bool(getattr(self, "noise_force_one", False))   # НЕ включай, если хочешь иногда “нулевую”

                    idict["img"], applied = apply_noise_recipe(
                        idict["img"],
                        cfg=cfg,
                        p_none=p_none,
                        p_boost=p_boost,
                        strength=strength,
                        force_at_least_one=force_one,
                    )

                    if debug_regions:
                        print("[render_text] noise applied:", applied)

                except Exception as e:
                    if debug_regions:
                        print("[render_text] post-noise error:", repr(e))
                    pass

            if viz:
                try:
                    if idict['wordBB'] is not None:
                        viz_textbb(1, idict['img'], [idict['wordBB']], alpha=1.0)
                    else:
                        import matplotlib.pyplot as plt
                        plt.figure(1); plt.clf()
                        plt.imshow(_rgb(idict['img']))
                        plt.axis('off')
                        _plt_stable_draw(plt.gcf(), pause=0.35)
                    viz_masks(2, idict['img'], seg_i, depth_f, regions['label'])
                except Exception:
                    pass

            res.append(idict.copy())

        return res
