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

# ==== DEBUG UTILS (вставить рядом с импортами, один раз) ====
# ==== DEBUG UTILS ====
import os, time, cv2

import numpy as np


import cv2
import numpy as np
import os

def save_result(text_mask, text, bbs, output_dir=r"C:\code\res"):
    """
    Сохраняет изображение с наложенным текстом и его границами в указанный каталог.
    
    Parameters:
    - text_mask: изображение с наложенным текстом (в черно-белом формате)
    - text: текст, который был наложен на изображение
    - bbs: границы (bounding boxes) текста
    - output_dir: путь к каталогу для сохранения изображения
    """
    # Проверка наличия каталога для сохранения
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Проверяем, что текстовая маска не пуста
    if text_mask is None or text_mask.size == 0:
        return
    
    # Если изображение в черно-белом формате, преобразуем в цветное для сохранения
    if len(text_mask.shape) == 2:  # Черно-белое изображение (H, W)
        color_img = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2RGB)
    else:  # Если уже цветное, оставляем как есть
        color_img = text_mask

    # Отображаем прямоугольники вокруг текста (bounding boxes)
    for bb in bbs:
        x, y, w, h = bb
        cv2.rectangle(color_img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # Формируем имя файла для сохранения
    file_name = f"generated_text_{text[:10]}..._bbox.png"
    file_path = os.path.join(output_dir, file_name)

    # Сохраняем изображение
    cv2.imwrite(file_path, color_img)


import numpy as np
import cv2

def depth_to_z(depth, near=1.0, far=5.0, invert=True):
    """
    Перевод произвольной карты глубины в Z (условные метры).
    depth: HxW, float
    near, far: диапазон расстояний камеры
    invert=True: для MiDaS (где ближе = больше значение) обычно True.
    """
    d = depth.astype(np.float32)
    d_min = float(d.min())
    d_max = float(d.max())
    if d_max - d_min < 1e-6:
        return np.full_like(d, (near + far) * 0.5, dtype=np.float32)

    d = (d - d_min) / (d_max - d_min + 1e-6)  # [0,1]
    if invert:
        d = 1.0 - d

    Z = near + d * (far - near)
    return Z


def euler_to_R(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """
    Матрица поворота из углов Эйлера (в градусах):
    yaw  - вокруг Y,
    pitch- вокруг X,
    roll - вокруг Z.
    Порядок: R = Rz(roll) * Ry(yaw) * Rx(pitch)
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ], dtype=np.float32)

    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0,          1, 0         ],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0,             0,            1],
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx
    return R


def warp_perspective_with_depth(
    rgb,
    depth,
    yaw_deg=0.0,
    pitch_deg=10.0,
    roll_deg=0.0,
    fov_deg=60.0,
    depth_near=1.0,
    depth_far=5.0,
    invert_depth=True,
):
    """
    Добавляет перспективный эффект к изображению, используя карту глубины.

    rgb   : HxWx3 или WxHx3, uint8 (RGB/BGR — геометрии всё равно)
    depth : HxW, float32
    """
    import numpy as np
    import cv2

    rgb = np.asarray(rgb)
    depth = np.asarray(depth)

    # --- depth приводим к 2D HxW и считаем эталонным размером ---
    if depth.ndim == 3:
        depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth должен быть 2D, а не {depth.shape}")

    H_d, W_d = depth.shape

    # --- rgb к 3 каналам ---
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb должен быть HxWx3, а не {rgb.shape}")

    H_i, W_i = rgb.shape[:2]

    # --- Согласование размеров rgb и depth ---
    if (H_i, W_i) != (H_d, W_d):
        # Классический pygame-кейс: rgb (W, H, 3), depth (H, W)
        if (W_i, H_i) == (H_d, W_d):
            # приводим rgb к (H, W, 3)
            rgb = np.swapaxes(rgb, 0, 1).copy()
            H_i, W_i = rgb.shape[:2]
        else:
            raise ValueError(
                f"warp_perspective_with_depth: несовместимые формы "
                f"rgb={rgb.shape}, depth={depth.shape}"
            )

    H, W = H_d, W_d
    cx = W * 0.5
    cy = H * 0.5

    # --- фокус по FOV ---
    f = 0.5 * W / np.tan(np.deg2rad(fov_deg) * 0.5)
    fx = fy = float(f)

    # --- Z из глубины ---
    Z = depth_to_z(depth, near=depth_near, far=depth_far, invert=invert_depth)

    # --- сетка пикселей ---
    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )

    # --- из пикселей в 3D ---
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).T  # (3, N)

    # --- поворот сцены ---
    R = euler_to_R(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)
    pts2 = R @ pts
    X2, Y2, Z2 = pts2[0, :], pts2[1, :], pts2[2, :]

    eps = 1e-4
    Z2 = np.maximum(Z2, eps)

    # --- обратно в пиксели ---
    u2 = fx * (X2 / Z2) + cx
    v2 = fy * (Y2 / Z2) + cy

    map_x = u2.reshape(H, W).astype(np.float32)
    map_y = v2.reshape(H, W).astype(np.float32)

    warped = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped




def get_placeable_mask(seg, min_area=100):
    """
    Создает маску, в которой текст может быть размещен на основе сегментации
    seg: Сегментированное изображение
    min_area: Минимальная площадь области, в которой можно разместить текст
    """
    mask = np.zeros_like(seg, dtype=np.uint8)
    
    # Перебираем все объекты в сегменте
    for label in np.unique(seg):
        if label == 0:
            continue  # Пропускаем фон
        
        # Выбираем все пиксели, которые принадлежат текущему объекту
        current_mask = (seg == label).astype(np.uint8)

        # Применяем фильтрацию по площади, чтобы исключить слишком маленькие области
        if np.sum(current_mask) >= min_area:
            mask = cv2.bitwise_or(mask, current_mask)  # Объединяем с общей маской
    
    return mask

def _dbg_dir():
    d = "debug_steps"; os.makedirs(d, exist_ok=True); return d

def _stamp():
    return time.strftime("%Y%m%d-%H%M%S")

def _save_img(name, img, assume_rgb=True):
    p = os.path.join(_dbg_dir(), name)
    if img is None: return
    if img.ndim == 2:
        cv2.imwrite(p, img)
    elif img.ndim == 3 and img.shape[2] == 3:
        out = to_bgr(img) if assume_rgb else img
        cv2.imwrite(p, out)
    else:
        import numpy as _np
        arr = img
        if arr.dtype != _np.uint8:
            arr = _np.clip(arr, 0, 255).astype(_np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            out = to_bgr(arr) if assume_rgb else arr
            cv2.imwrite(p, out)
        else:
            cv2.imwrite(p, arr)

def mean_color_under_mask(rgb, mask):
    """Средний цвет под бинарной маской (в RGB 0–1)."""
    if mask.dtype != np.bool_:
        mask = mask > 0
    if mask.sum() == 0:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    rgbf = rgb.astype(np.float32) / 255.0
    mean = rgbf[mask].mean(axis=0)
    return mean

def relative_luminance(rgb):
    """Яркость по sRGB."""
    r, g, b = rgb
    return 0.2126*r + 0.7152*g + 0.0722*b

def color_contrast(a, b):
    """Разница по яркости."""
    return abs(relative_luminance(a) - relative_luminance(b))

def choose_contrast_color(bg_rgb, min_contrast=0.4):
    """
    Возвращает цвет текста (RGB 0–1), контрастный к bg_rgb и достаточно яркий.

    bg_rgb: np.array длиной 3 в диапазоне [0,1].
    """
    import numpy as np
    import colorsys

    bg = np.asarray(bg_rgb, dtype=np.float32).clip(0.0, 1.0)

    # Пытаемся подобрать яркий цвет в HSV с хорошей насыщенностью/яркостью
    for _ in range(50):
        h = np.random.rand()
        s = np.random.uniform(0.6, 1.0)   # насыщенный
        v = np.random.uniform(0.8, 1.0)   # яркий
        txt_rgb = np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)

        if color_contrast(bg, txt_rgb) >= min_contrast:
            return txt_rgb

    # fallback — инвертируем фон
    inv = 1.0 - bg
    if color_contrast(bg, inv) < min_contrast:
        # максимально «жёсткий» контраст: чёрно-белое по компонентам
        inv = np.where(bg < 0.5, 1.0, 0.0).astype(np.float32)
    return invсв

def _save_mask(name, m):
    import numpy as _np
    if m is None: return
    mm = (m > 0).astype(_np.uint8) * 255
    _save_img(name, mm)

def _bbox_from_mask_uint(mask_uint8):
    import numpy as _np
    if mask_uint8 is None: return None
    ys, xs = _np.where(mask_uint8 > 0)
    if xs.size == 0 or ys.size == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _draw_rect(img, bbox, color=(255, 0, 0), th=2):
    if bbox is None: return img
    x0, y0, x1, y1 = bbox
    out = img.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), color, th)
    return out

# ==== FAST FIT UTILS (coarse-to-fine) ====
def ii_from_free(free_bin255):
    free01 = (free_bin255 > 0).astype(np.uint8)
    return cv2.integral(free01, sdepth=cv2.CV_32S)

def rect_sum(ii, y0, x0, h, w):
    y1, x1 = y0 + h, x0 + w
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

def any_fit_with_integral(free_bin255, need_h, need_w, step=3):
    """
    Проверка, можно ли вписать прямоугольник need_h x need_w в free_bin255
    (0/255, где >0 = свободно), с шагом step по координатам.
    Логика та же, что была, но без питоновских двойных циклов.
    """
    import numpy as np
    import cv2

    # 0/1 карта "свободных" ячеек
    free01 = (free_bin255 > 0).astype(np.uint8)
    H, W = free01.shape[:2]
    if need_h > H or need_w > W:
        return False, None

    # Интегральное изображение: (H+1, W+1)
    ii = cv2.integral(free01, sdepth=cv2.CV_32S)

    hh = H - need_h + 1
    ww = W - need_w + 1
    if hh <= 0 or ww <= 0:
        return False, None

    area = need_h * need_w

    # Суммы по всем окнам размера need_h x need_w:
    # S[y,x] = ii[y+need_h, x+need_w] - ii[y, x+need_w] - ii[y+need_h, x] + ii[y, x]
    A = ii[need_h:, need_w:]
    B = ii[:-need_h, need_w:]
    C = ii[need_h:, :-need_w]
    D = ii[:-need_h, :-need_w]
    sums = A - B - C + D  # shape = (hh, ww)

    # Учитываем шаг по координатам (как раньше step в цикле)
    sums_sub = sums[0:hh:step, 0:ww:step]
    ys, xs = np.where(sums_sub == area)
    if ys.size == 0:
        return False, None

    # Берём первую найденную позицию (раньше тоже "первая по порядку")
    y0 = int(ys[0] * step)
    x0 = int(xs[0] * step)
    return True, (y0, x0)


def downscale_mask(m, scale=0.5):
    H, W = m.shape[:2]
    return cv2.resize(m, (max(1, int(W*scale)), max(1, int(H*scale))), interpolation=cv2.INTER_NEAREST)

def fits_coarse_to_fine(place_mask, need_h, need_w,
                        coarse_scale=0.5, coarse_step=4, fine_step=2):
    """
    Двухуровневая проверка (coarse → fine), как раньше.
    place_mask: 0 = можно, 255 = нельзя.
    """
    # Оставляем существующее поведение (эти три строки и раньше всегда включали FAST)
    coarse_scale = 0.33 if 'FAST' else coarse_scale
    coarse_step  = 8     if 'FAST' else coarse_step
    fine_step    = 3     if 'FAST' else fine_step

    import numpy as np

    # --- coarse уровень ---
    pm_s = downscale_mask(place_mask, coarse_scale)
    free_small = (pm_s == 0).astype(np.uint8) * 255

    ok_c, _ = any_fit_with_integral(
        free_small,
        max(1, int(need_h * coarse_scale)),
        max(1, int(need_w * coarse_scale)),
        step=coarse_step,
    )
    if not ok_c:
        return False

    # --- fine уровень ---
    free_full = (place_mask == 0).astype(np.uint8) * 255
    ok_f, _ = any_fit_with_integral(
        free_full,
        need_h,
        need_w,
        step=fine_step,
    )
    return ok_f

def score_region_fast(place_mask):
    return int(cv2.countNonZero((place_mask == 0).astype(np.uint8) * 255))

def select_topK_regions(place_masks, K=6):
    scores = [(score_region_fast(pm), i) for i, pm in enumerate(place_masks)]
    scores.sort(reverse=True)
    K_eff = 2 if getattr(globals().get('self', None), 'fast_mode', False) else K  # безопасно
    return [i for _, i in scores[:min(K_eff, len(scores))]]

def build_forbid_adaptive(place_mask: np.ndarray, f_h_px: float, gap_px: int = 6):
    gap = int(min(gap_px, max(1, 0.6 * float(f_h_px))))
    k = max(1, (gap // 2) * 2 + 1)  # нечётный размер ядра
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    forbid = cv2.dilate(place_mask, kernel, iterations=1)
    return forbid

def estimate_local_scale(Hinv, yx_fp, delta: int = 4):
    """
    Грубая оценка масштаба «фронто→изображение» в точке (y,x) FP:
    берём маленький крестик (±delta) и смотрим длины после варпа.
    Возвращает минимальный из масштабов по осям (чтобы быть консервативнее).
    """
    import numpy as np, cv2
    y, x = float(yx_fp[0]), float(yx_fp[1])
    pts = np.array([
        [x,     y    , 1.0],
        [x+delta, y  , 1.0],
        [x,     y+delta, 1.0],
    ], dtype=np.float32).T  # 3x3
    wpts = Hinv.dot(pts)
    wpts /= (wpts[2:3, :] + 1e-6)
    p0 = wpts[:2, 0]
    px = wpts[:2, 1]
    py = wpts[:2, 2]
    sx = np.linalg.norm(px - p0) / max(delta, 1e-6)
    sy = np.linalg.norm(py - p0) / max(delta, 1e-6)
    return float(min(sx, sy))

def estimate_text_box(txt: str, f_h_px: float, f_asp: float,
                      space_factor: float = SPACE_FACTOR):
    """Оценка (need_h, need_w) для ОДНОСТРОЧНОГО текста."""
    chars = len(txt.replace("\n", ""))
    spaces = txt.count(' ')
    char_w = f_h_px * f_asp
    need_w = chars * char_w + spaces * (space_factor * char_w)
    need_h = f_h_px
    return int(np.ceil(need_h)), int(np.ceil(need_w))

def region_can_fit_rect(forbidden_mask: np.ndarray, need_h: int, need_w: int, step: int = 3):
    """Версия на интегральных изображениях: O(1) на окно, скан с шагом."""
    free255 = (forbidden_mask == 0).astype(np.uint8) * 255
    ok, _ = any_fit_with_integral(free255, need_h, need_w, step=step)
    return ok, 1 if ok else 0  # score больше не считаем — ранний выход



import numpy as np

def _warp_points(Hinv, pts_xy):
    """pts_xy: (N,2) в FP -> (N,2) в IMG"""
    pts = np.concatenate([pts_xy.astype(np.float32), np.ones((len(pts_xy),1), np.float32)], axis=1).T  # 3xN
    w = Hinv @ pts
    w /= (w[2:3, :] + 1e-6)
    return w[:2, :].T

def estimate_local_scale_grid(Hinv, mask_fp, k: int = 9, delta: int = 6):
    """
    Оценивает местный масштаб (фронто->изображение) стабильно:
    - берём k точек внутри маски (равномерно по bbox);
    - для каждой считаем sx, sy по сдвигам ±delta;
    - возвращаем p25 по min(sx,sy) среди точек.
    """
    ys, xs = np.where(mask_fp > 0)
    if ys.size == 0:
        return 0.0

    # равномерно берём до k точек внутри bbox маски
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    gy = np.linspace(y0, y1, num=int(np.ceil(np.sqrt(k))), dtype=np.int32)
    gx = np.linspace(x0, x1, num=int(np.ceil(np.sqrt(k))), dtype=np.int32)
    grid = np.stack(np.meshgrid(gy, gx, indexing='ij'), axis=-1).reshape(-1, 2)
    grid = grid[:k]
    # фильтр по реальной маске
    msel = (mask_fp[grid[:,0], grid[:,1]] > 0)
    pts = grid[msel]
    if len(pts) == 0:
        # fallback — центр маски
        cy, cx = int(np.median(ys)), int(np.median(xs))
        pts = np.array([[cy, cx]], dtype=np.int32)

    scales = []
    for (yy, xx) in pts:
        probe = np.array([[xx, yy],
                          [xx+delta, yy],
                          [xx, yy+delta]], dtype=np.float32)
        w = _warp_points(Hinv, probe)
        p0, px, py = w[0], w[1], w[2]
        sx = np.linalg.norm(px - p0) / max(delta, 1e-6)
        sy = np.linalg.norm(py - p0) / max(delta, 1e-6)
        scales.append(min(sx, sy))
    if len(scales) == 0:
        return 0.0
    return float(np.percentile(scales, 25))  # p25 вместо минимума


# ============================================================


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
            xs, ys = np.where(mask)

            coords = np.c_[xs, ys].astype('float32')
            rect = cv2.minAreaRect(coords)
            # box = np.array(cv2.cv.BoxPoints(rect))
            box = np.array(cv2.boxPoints(rect))
            h, w, rot = TextRegions.get_hw(box, return_rot=True)

            # --- мягкий фильтр по аспект-ратио (как раньше, но порог выше) ---
            # Очень вытянутые регионы (узкие полоски / линии) отбрасываем.
            # aspect >= 1, чем больше — тем более вытянутый регион.
            aspect = max(float(h) / max(float(w), 1.0),
                         float(w) / max(float(h), 1.0))

            rect_area = max(1.0, float(w) * float(h))
            f = (
                h > TextRegions.minHeight * 0.8 and          # немного смягчили порог
                w > TextRegions.minWidth * 0.8 and
                (float(area[idx]) / rect_area) >= (TextRegions.pArea * 0.85) and  # чуть мягче
                aspect < 18.0                                # <-- мягкий порог по аспект-ратио
            )
            filt.append(f)
            R.append(rot)

        # filter bad regions:
        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        aidx = np.argsort(-area)

        # --- НОВОЕ: ограничиваем число регионов, с которыми пойдём дальше ---
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
    def sample_grid_neighbours(mask, nsample, step=3):
        """
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.
        """
        if 2 * step >= min(mask.shape[:2]):
            return  # None

        y_m, x_m = np.where(mask)
        mask_idx = np.zeros_like(mask, 'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i], x_m[i]] = i

        xp, xn = np.zeros_like(mask), np.zeros_like(mask)
        yp, yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:, :-2 * step] = mask[:, 2 * step:]
        xn[:, 2 * step:] = mask[:, :-2 * step]
        yp[:-2 * step, :] = mask[2 * step:, :]
        yn[2 * step:, :] = mask[:-2 * step, :]
        valid = mask & xp & xn & yp & yn

        ys, xs = np.where(valid)
        N = len(ys)
        if N == 0:  # no valid pixels in mask:
            return  # None
        nsample = min(nsample, N)
        idx = np.random.choice(N, nsample, replace=False)
        # generate neighborhood matrix:
        # (1+4)x2xNsample (2 for y,x)
        xs, ys = xs[idx], ys[idx]
        s = step
        X = np.transpose(np.c_[xs, xs + s, xs + s, xs - s, xs - s][:, :, None], (1, 2, 0))
        Y = np.transpose(np.c_[ys, ys + s, ys - s, ys + s, ys - s][:, :, None], (1, 2, 0))
        sample_idx = np.concatenate([Y, X], axis=1)
        mask_nn_idx = np.zeros((5, sample_idx.shape[-1]), 'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:, i] = mask_idx[sample_idx[:, :, i][:, 0], sample_idx[:, :, i][:, 1]]
        return mask_nn_idx

    @staticmethod
    def filter_depth(xyz, seg, regions, max_planes=6):
        """
        Выбираем несколько (до max_planes) планарных регионов, стараясь
        отбрасывать "небо", но делаем RANSAC мягче.

        regions: словарь после get_regions:
          - 'label': метки сегментов
          - 'area' : площади
          - 'rot'  : повороты (опц.)

        Возвращает словарь с полями:
          'label', 'area', 'coeff', 'inliers', 'rot'
        — по структуре совместим с остальным кодом.
        """
        import numpy as np

        xyz = np.asarray(xyz, dtype=np.float32)
        seg = np.asarray(seg, dtype=np.int32)

        labels = np.asarray(regions.get("label", []), dtype=np.int32)
        areas = np.asarray(regions.get("area", []), dtype=np.float32)
        rots = regions.get("rot", [None] * len(labels))

        plane_info = {
            'label': [],
            'coeff': [],
            'inliers': [],
            'area': [],
            'rot': [],
        }

        if labels.size == 0:
            return plane_info

        # --- сортируем по площади: сначала самые большие ---
        order = np.argsort(-areas)
        labels = labels[order]
        areas = areas[order]
        rots = [rots[i] for i in order]

        H, W = seg.shape[:2]
        total_px = float(H * W)

        def is_sky_like(lbl, area_px):
            """Эвристика: большой, верхний, широкий регион = похоже на небо."""
            mask = (seg == int(lbl))
            ys, xs = np.where(mask)
            if ys.size == 0:
                return False

            area_frac = float(area_px) / max(total_px, 1.0)

            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            h_frac = float(y_max - y_min + 1) / max(H, 1)
            w_frac = float(x_max - x_min + 1) / max(W, 1)
            y_center_frac = (0.5 * (y_min + y_max)) / max(H, 1)

            cond_area = area_frac >= 0.18      # достаточно большой кусок
            cond_top = y_center_frac <= 0.38   # явно верх кадра
            cond_wide = w_frac >= 0.45         # широкий по X
            cond_not_too_tall = h_frac <= 0.75 # не во всю высоту

            if cond_area and cond_top and cond_wide and cond_not_too_tall:
                print(
                    f"[filter_depth] skip label={int(lbl)} as sky-like: "
                    f"area={area_frac:.3f}, y_center={y_center_frac:.2f}, "
                    f"w={w_frac:.2f}, h={h_frac:.2f}"
                )
                return True
            return False

        # --- НОВОЕ: ограничение на число регионов, для которых вообще запускаем RANSAC ---
        max_trials = getattr(TextRegions, "maxPlaneTrials", None)

        # --- перебираем регионы по убыванию площади ---
        for idx, (lbl, a, r) in enumerate(zip(labels, areas, rots)):
            if len(plane_info['label']) >= max_planes:
                break

            if (max_trials is not None) and (idx >= int(max_trials)):
                print(f"[filter_depth] reached maxPlaneTrials={max_trials}, stop scanning regions")
                break

            # сначала выкидываем явно "небоподобные" регионы
            if is_sky_like(lbl, a):
                continue

            reg_mask = (seg == int(lbl))
            if not np.any(reg_mask):
                continue

            pt = xyz[reg_mask]      # (N, 3)
            n_pt = pt.shape[0]

            # более мягкое условие по числу точек
            if n_pt < 25:
                # совсем микроскопические регионы не трогаем
                continue

            # соседства для RANSAC: сначала по сетке, если не получилось — рандом
            nn_idx = TextRegions.sample_grid_neighbours(
                reg_mask,
                TextRegions.ransac_fit_trials
            )
            if nn_idx is None:
                nn_idx = np.random.randint(
                    0, n_pt,
                    size=(5, TextRegions.ransac_fit_trials),
                    dtype=np.int32
                )

            # адаптивное число инлаеров: не фиксированное, а доля от n_pt
            min_inlier = max(15, int(0.30 * n_pt))  # ≥15, но не требуем половину точек

            plane_model = su.isplanar(
                pt,
                nn_idx,
                TextRegions.dist_thresh,
                min_inlier,
                TextRegions.min_z_projection
            )

            # fallback: если RANSAC не смог — пробуем LS-плоскость по всем точкам
            if plane_model is None:
                print(f"[filter_depth] RANSAC failed for label={int(lbl)}, trying LS fallback")
                try:
                    X = np.c_[pt[:, 0], pt[:, 1], np.ones(pt.shape[0])]
                    y = -pt[:, 2]
                    coeff_ls, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    a_c, b_c, d_c = coeff_ls
                    c_c = 1.0
                    coeff = np.array([a_c, b_c, c_c, d_c], dtype=np.float32)
                    inliers = np.arange(pt.shape[0], dtype=np.int32)
                    plane_model = (coeff, inliers)
                    print(f"[filter_depth] LS-plane accepted for label={int(lbl)}")
                except Exception as e:
                    print(f"[filter_depth] LS fallback failed for label={int(lbl)}:", repr(e))
                    continue

            coeff, inliers = plane_model

            # мягкий фильтр по нормали: только логируем очень "плоские" по Z плоскости,
            # но не отбрасываем их жёстко
            if np.abs(coeff[2]) <= (TextRegions.min_z_projection * 0.5):
                print(
                    f"[filter_depth] label={int(lbl)} has weak z-normal: "
                    f"coeff={coeff}"
                )
                # НЕ делаем continue — всё равно принимаем плоскость

            # успешный кандидат — сохраняем, но НЕ выходим из цикла
            plane_info['label'].append(int(lbl))
            plane_info['coeff'].append(np.asarray(coeff, dtype=np.float32))
            plane_info['inliers'].append(inliers)
            plane_info['area'].append(float(a))
            plane_info['rot'].append(r)

            print(
                f"[filter_depth] accepted label={int(lbl)}, "
                f"area={float(a):.1f}, n_pt={n_pt}, "
                f"min_inlier={min_inlier}, total_kept={len(plane_info['label'])}"
            )

        # приведение к массивам
        if plane_info['coeff']:
            plane_info['label'] = np.asarray(plane_info['label'], dtype=np.int32)
            plane_info['area'] = np.asarray(plane_info['area'], dtype=np.float32)
            plane_info['coeff'] = np.asarray(plane_info['coeff'], dtype=np.float32)
            # rot оставляем списком — filter_for_placement умеет с этим жить
        else:
            print("[filter_depth] no planar regions kept (even with softened RANSAC)")
            plane_info['label'] = np.zeros((0,), dtype=np.int32)
            plane_info['area'] = np.zeros((0,), dtype=np.float32)
            plane_info['coeff'] = np.zeros((0, 4), dtype=np.float32)
            plane_info['rot'] = []

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


def get_text_placement_mask(xyz,mask,plane,pad=2,viz=False):
    contour,hier = cv2.findContours(mask.copy().astype('uint8'),
                                    mode=cv2.RETR_CCOMP,
                                    method=cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contour = [np.squeeze(c).astype('float') for c in contour]
    H,W = mask.shape[:2]
    
    # bring the contour 3d points to fronto-parallel config:
    pts,pts_fp = [],[]
    center = np.array([W,H])/2
    n_front = np.array([0.0,0.0,-1.0])
    for i in range(len(contour)):
        cnt_ij = contour[i]
        xyz = su.DepthCamera.plane2xyz(center, cnt_ij, plane)
        R = su.rot3d(plane[:3],n_front)
        xyz = xyz.dot(R.T)
        pts_fp.append(xyz[:,:2])
        pts.append(cnt_ij)

    # unrotate in 2D plane:
    rect = cv2.minAreaRect(pts_fp[0].copy().astype('float32'))
    box = np.array(cv2.boxPoints(rect))
    R2d = su.unrotate2d(box.copy())
    box = np.vstack([box,box[0,:]]) # close for viz

    mu = np.median(pts_fp[0],axis=0)
    pts_tmp = (pts_fp[0]-mu[None,:]).dot(R2d.T) + mu[None,:]
    boxR = (box-mu[None,:]).dot(R2d.T) + mu[None,:]

    # rescale to approx target region:
    s = rescale_frontoparallel(pts_tmp,boxR,pts[0])
    boxR *= s
    for i in range(len(pts_fp)):
        pts_fp[i] = s*((pts_fp[i]-mu[None,:]).dot(R2d.T) + mu[None,:])

    # paint the unrotated contour points:
    minxy = -np.min(boxR,axis=0)
    ROW = np.max(ssd.pdist(np.atleast_2d(boxR[:,0]).T))
    COL = np.max(ssd.pdist(np.atleast_2d(boxR[:,1]).T))

    # <<< ключевая правка: даем больше «полотна» и полей >>>
    ROW *= 1.12   # +12% ширины
    COL *= 1.06   # +6% высоты
    pad = max(int(pad), 14)

    place_mask = 255*np.ones((int(np.ceil(COL))+pad, int(np.ceil(ROW))+pad), 'uint8')

    pts_fp_i32 = [(pts_fp[i]+(minxy + pad//2)[None,:]).astype('int32') for i in range(len(pts_fp))]
    cv2.drawContours(place_mask, pts_fp_i32, -1, 0, thickness=cv2.FILLED, lineType=8, hierarchy=hier)

    if not TextRegions.filter_rectified((~place_mask).astype('float')/255):
        return

    # calculate the homography
    H,_ = cv2.findHomography(pts[0].astype('float32').copy(),
                             pts_fp_i32[0].astype('float32').copy(), method=0)
    Hinv,_ = cv2.findHomography(pts_fp_i32[0].astype('float32').copy(),
                                pts[0].astype('float32').copy(), method=0)

    if viz:
        plt.subplot(1,2,1); plt.imshow(mask)
        plt.subplot(1,2,2); plt.imshow(~place_mask)
        for i in range(len(pts_fp_i32)):
            plt.scatter(pts_fp_i32[i][:,0],pts_fp_i32[i][:,1],
                        edgecolors='none',facecolor='g',alpha=0.5)
        plt.show()

    return place_mask, H, Hinv


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

def to_bgr(arr):
    import numpy as _np, cv2 as _cv2
    if arr is None: return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)
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
    import cv2, numpy as np
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

def build_forbidden_mask_rectified(place_mask, H, occupied_global, depth, gap=3):
    """
    place_mask: uint8, 255 — нельзя, 0 — можно (в координатах выпрямленного региона)
    H: гомография image->rectified
    occupied_global: uint8, 255 — уже занято (в IMG)
    depth: float карта глубины в IMG
    """
    forbidden = place_mask.copy()
    Hh, Hw = place_mask.shape[:2]
    # Уже занятые области -> в координаты региона + мягкая дилатация
    if occupied_global is not None and np.any(occupied_global):
        occ_rect = cv2.warpPerspective(occupied_global, H, (Hw, Hh),
                                       flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap+1, 2*gap+1))
        occ_rect = cv2.dilate(occ_rect, kernel, 1)
        forbidden = np.maximum(forbidden, occ_rect)

    # Край региона — тонкая рамка, затем небольшая дилатация
    allowed = (place_mask == 0).astype(np.uint8) * 255
    k_edge = np.ones((3,3), np.uint8)
    border = cv2.morphologyEx(allowed, cv2.MORPH_GRADIENT, k_edge)
    border = cv2.dilate(border, np.ones((max(1, 2*gap-1), max(1, 2*gap-1)), np.uint8), 1)
    forbidden = np.maximum(forbidden, border)

    # Резкие перепады глубины в координатах региона
    if depth is not None:
        depth_rect = cv2.warpPerspective(depth.astype(np.float32), H, (Hw, Hh),
                                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        dzdx = cv2.Sobel(depth_rect, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(depth_rect, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(dzdx, dzdy)
        if np.any(np.isfinite(grad)):
            tau = np.percentile(grad[np.isfinite(grad)], 85)
            depth_edges = (grad > tau).astype(np.uint8) * 255
            depth_edges = cv2.dilate(depth_edges, np.ones((2*gap+1, 2*gap+1), np.uint8), 1)
            forbidden = np.maximum(forbidden, depth_edges)

    return forbidden


class RendererV3(object):

    def __init__(self, data_dir, max_time=None):
        self.debug_bbox = True        # контур bbox
        self.debug_bg = True         # полупрозрачная подложка
        self.debug_save_mask = False
        self.text_renderer = tu.RenderFont(data_dir)
        self.colorizer = Colorize(data_dir)
        #self.colorizerV2 = colorV2.Colorize(data_dir)

        self.min_box_gap_rect_px = 30  # было 6
        self.min_box_gap_px = 12

        self.max_text_regions = 7

        self.max_time = max_time

        self.global_attempt_budget = 40      # максимум попыток текста на инстанс
        self.per_region_attempt_cap = 6      # максимум попыток на один регион
        self.reject_patience_tiny = 5        # было 0.10
        self.accept_min_area_img_px_override = 120  # НОВОЕ: абсолютный проход по площади маски

        self.target_min_img_side = 10  # новая настройка для предсказания

        # жесткие минимумы/максимумы
        self.global_attempt_budget_base = 10   # было 40
        self.per_region_attempt_cap_base = 3   # было 6
        self.max_shrink_trials_base = 3        # было 6

        # --- Runtime-атрибуты (ставятся в render_text) ---
        self._max_shrink_trials_runtime = self.max_shrink_trials_base

        # --- Кэш неудачных пар (регион, высота) ---
        self._failed_pairs = set()  # {(ireg:int, f_px:int)}

        self.min_word_len = 4   # минимальная длина алфанумерик-токена
        self.char_gap_px  = 0   # базовый межсимвольный зазор
        self.min_char_px_img = 7
        self.min_word_side_img = 7              # было 14
        self.accept_min_side_px_override = 6    # было 30
        self.min_fp2img_area_ratio = 0.08
        self.min_char_height = max(8, MIN_FONT_PX)     # было 12
        self.min_asp_ratio   = 0.30
        self.viz_fallback_cv = True

        self.no_geom = False

        self.fast_mode = True  # << включи/выключи ускорение одним флажком

        self.char_gap_rel = -0.5   # -15% к интервалу между ВСЕМИ символами
        self.char_gap_px  = 0       # можно 0
        self.word_gap_rel = 0.0     # пока без относительного эффекта по словам
        self.word_gap_px  = 2 

        self.enable_perspective_warp = True  # включать вручную в конфиге
        self.persp_yaw_range   = 10.0   # градусов влево/вправо
        self.persp_pitch_range = 6.0    # вверх/вниз
        self.persp_roll_range  = 3.0    # "кручение" вокруг оси
        self.persp_fov_deg     = 60.0
        self.persp_depth_near  = 1.0
        self.persp_depth_far   = 6.0
        self.persp_invert_depth = True

        self.min_text_rel_height = 0.06  # 4% от min(H,W)
        self.min_text_aspect = 2.5

        if self.fast_mode:
            # 1 блок текста на картинку, минимум попыток и ужиманий
            self.max_text_regions = 1
            self.global_attempt_budget_base = 5
            self.per_region_attempt_cap_base = 2
            self.max_shrink_trials_base = 2

            # поменьше «мелочи» — меньше неудачных масок
            self.min_word_len = 5
            self.min_char_px_img = 8
            self.min_word_side_img = 8
            self.accept_min_side_px_override = 7

            # жёстко глушим отладочный I/O
            self.debug_save_mask = False
            self.debug_bbox = False
            self.debug_bg = False

    def get_region_center_img(self, ireg, place_masks, regions, img_shape):
        """
        Центр региона (y, x) в КООРДИНАТАХ ИЗОБРАЖЕНИЯ.

        - При no_geom=True маска уже в image-space → берём центр по 0-пикселям.
        - При no_geom=False маска в фронто-параллельной системе:
          считаем центр там и проектируем через Hinv в image-space.
        """
        import numpy as np

        H_img, W_img = img_shape
        mask_fp = place_masks[ireg]

        ys, xs = np.where(mask_fp == 0)  # 0 = можно → реальный регион
        if ys.size == 0:
            return H_img // 2, W_img // 2

        cy_fp = float(ys.mean())
        cx_fp = float(xs.mean())

        use_geom = (not getattr(self, "no_geom", False)) \
                   and ("homography_inv" in regions) \
                   and (ireg < len(regions["homography_inv"]))

        if use_geom:
            Hinv = regions["homography_inv"][ireg]
            # _warp_points ожидает (N,2): (x,y)
            pts = np.array([[cx_fp, cy_fp]], dtype=np.float32)
            w = _warp_points(Hinv, pts)  # (N,2) в image-space
            cx_img = float(w[0, 0])
            cy_img = float(w[0, 1])
        else:
            cx_img, cy_img = cx_fp, cy_fp

        cy_img = int(max(0, min(H_img - 1, round(cy_img))))
        cx_img = int(max(0, min(W_img - 1, round(cx_img))))
        return cy_img, cx_img


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

    def filter_regions(self,regions,filt):
        """
        filt : boolean list of regions to keep.
        """
        idx = np.arange(len(filt))[filt]
        for k in regions.keys():
            regions[k] = [regions[k][i] for i in idx]
        return regions
    

    def filter_for_placement(self, xyz, seg, regions, viz=False):
        """
        Вариант максимально близкий к оригинальному SynthText:
        - для каждого планарного региона считаем fronto-parallel маску через get_text_placement_mask
        - сохраняем place_mask, H, Hinv в словарь regions
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

        for i in range(n):
            lbl = int(labels[i])

            # маска региона в исходном seg
            mask = (seg == lbl).astype("uint8")

            # коэффициенты плоскости ax + by + cz + d = 0
            if i >= len(coeffs):
                print(f"[filter_for_placement] region {i}, label={lbl}: coeff index out of range, skip")
                continue

            plane = coeffs[i]

            res = get_text_placement_mask(xyz, mask, plane, pad=2, viz=viz)
            if res is None:
                print(f"[filter_for_placement] region {i}, label={lbl}: get_text_placement_mask -> None (rejected)")
                continue

            place_mask_fp, H, Hinv = res

            if place_mask_fp is None or place_mask_fp.size == 0 or place_mask_fp.sum() < 50:
                print(f"[filter_for_placement] region {i}, label={lbl}: too small rectified mask, skip")
                continue

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

        out["place_mask"]     = place_masks           # список 2D масок FP
        out["homography"]     = homographies          # image -> rectified
        out["homography_inv"] = homographies_inv      # rectified -> image

        print(f"[filter_for_placement] done, kept {len(place_masks)} regions")
        return out


    
    def select_middle_regions_by_area(self, place_masks, K=6):
        """
        Теперь выбираем ТОП-K регионов по площади исходного сегмента
        (regions['area']), а не по площади free-area в place_mask.

        K – максимальное число регионов (обычно 4).
        """
        import numpy as np

        # Пытаемся использовать реальные площади регионов из последнего вызова
        regions = getattr(self, "_regions_last", None)

        if regions is not None and "area" in regions:
            areas = np.asarray(regions["area"], dtype=np.float32)
            if areas.size == 0:
                return []

            # сортируем по убыванию площади
            idx_sorted = np.argsort(-areas)
            K_eff = min(int(K), len(idx_sorted))
            top_idx = idx_sorted[:K_eff]

            print(
                "[select_regions] top-{} by region area: {}".format(
                    K_eff,
                    [(int(i), float(areas[i])) for i in top_idx]
                )
            )
            return [int(i) for i in top_idx]

        # --- Fallback: если по какой-то причине нет regions/area ---
        stats = []
        for i, pm in enumerate(place_masks):
            free_area = int(np.count_nonzero(pm == 0))  # 0 = можно
            stats.append((free_area, i))
        stats.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in stats[:K]]



    def get_dominant_text_orientation(self, img, region_mask=None):
        """
        Определяет доминирующую ориентацию текста на изображении в градусах.
        Если задана region_mask, анализируется только указанный регион.
        
        Возвращает угол в диапазоне [0, 360) градусов, где 0 - горизонтальный текст,
        положительные значения - поворот против часовой стрелки.
        """
        import cv2
        import numpy as np
        
        # Преобразуем в градации серого, если цветное изображение
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Применяем маску региона, если она задана
        if region_mask is not None:
            # Создаем копию, чтобы не изменять исходное изображение
            gray = gray.copy()
            # Делаем фон белым вне региона (текст обычно темный на светлом фоне)
            gray[region_mask != 0] = 255
        
        # Повышаем контраст для лучшего обнаружения текста
        gray = cv2.equalizeHist(gray)
        
        # Бинаризация изображения
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Находим контуры текстовых областей
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0  # Если контуры не найдены, возвращаем горизонтальную ориентацию
        
        # Фильтруем маленькие контуры (шум)
        min_contour_area = 50
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        if not filtered_contours:
            return 0
        
        # Собираем все точки контуров для анализа
        all_points = []
        for cnt in filtered_contours:
            all_points.extend(cnt.reshape(-1, 2))
        
        if len(all_points) < 10:
            return 0
        
        # Преобразуем в numpy массив
        points = np.array(all_points, dtype=np.float32)
        
        # Применяем PCA для нахождения главных направлений
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Главный вектор - соответствует наибольшему собственному значению
            dominant_vector = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Вычисляем угол наклона
            angle_rad = np.arctan2(dominant_vector[1], dominant_vector[0])
            angle_deg = np.degrees(angle_rad)
            
            # Преобразуем в диапазон [0, 360)
            if angle_deg < 0:
                angle_deg += 360
            
            # Нормализуем относительно горизонтали (0 градусов)
            # Текст обычно имеет ориентацию близкую к 0 или 90 градусам
            normalized_angle = angle_deg % 180  # Учитываем симметрию текста
            
            return normalized_angle
        except:
            return 0

    def estimate_text_size(self, txt_str, font_size, aspect_ratio, angle=0):
        """
        Оценивает размеры текста с учетом возможного поворота.
        
        txt_str: строка текста
        font_size: размер шрифта
        aspect_ratio: соотношение сторон шрифта
        angle: угол поворота в градусах
        
        Возвращает: (height, width) - оценочные размеры текста
        """
        # Оцениваем базовые размеры без поворота
        base_height, base_width = self.calculate_text_box_size(txt_str, font_size, aspect_ratio)
        base_height, base_width = self.apply_padding(base_height, base_width)
        
        if abs(angle) < 1e-3:
            return base_height, base_width
        
        # Для поворота вычисляем новые размеры
        angle_rad = np.deg2rad(angle)
        new_width = abs(base_width * np.cos(angle_rad)) + abs(base_height * np.sin(angle_rad))
        new_height = abs(base_width * np.sin(angle_rad)) + abs(base_height * np.cos(angle_rad))
        
        return int(new_height), int(new_width)
    
    def get_text_center_position(self, region_coords, text_height, text_width):
        """
        Определяет центр будущего текста внутри региона.
        
        region_coords: (y_min, x_min, height, width) - координаты региона
        text_height, text_width: размеры текста после поворота
        
        Возвращает: (y_center_text, x_center_text) - координаты центра текста
        """
        y_min, x_min, reg_h, reg_w = region_coords
        
        # Вычисляем допустимые границы для размещения
        max_y_start = y_min + reg_h - text_height
        min_y_start = y_min
        max_x_start = x_min + reg_w - text_width
        min_x_start = x_min
        
        # Проверяем, что текст помещается в регион
        if min_y_start > max_y_start or min_x_start > max_x_start:
            # Если не помещается, используем центр региона как fallback
            y_center = y_min + reg_h // 2
            x_center = x_min + reg_w // 2
            return y_center, x_center
        
        # Выбираем позицию с приоритетом к центру региона (для более естественного размещения)
        y_center_reg = y_min + reg_h // 2
        x_center_reg = x_min + reg_w // 2
        
        # Вычисляем оптимальную позицию, стремясь к центру региона
        y0 = max(min_y_start, min(max_y_start, y_center_reg - text_height // 2))
        x0 = max(min_x_start, min(max_x_start, x_center_reg - text_width // 2))
        
        # Вычисляем центр текста
        y_center_text = y0 + text_height // 2
        x_center_text = x0 + text_width // 2
        
        return y_center_text, x_center_text
    
    def get_direction_to_center_for_text(self, text_center_y, text_center_x, img_shape):
        """
        Вычисляет направление ОТ центра изображения К центру текста.
        Угол в градусах, 0° — вправо, 90° — вниз, [0, 360).
        """
        import numpy as np

        H, W = img_shape
        img_center_y = H // 2
        img_center_x = W // 2

        dy = float(text_center_y) - float(img_center_y)
        dx = float(text_center_x) - float(img_center_x)

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        if angle_deg < 0.0:
            angle_deg += 360.0

        return angle_deg



    def select_optimal_angle(self, candidate_angles, dominant_orientation, zone_type, 
                             region_coords, img_shape, text_height, text_width):
        """
        Выбирает оптимальный угол для текста.

        На боковых сегментах ('horizontal'):
          - строим линию ОТ центра изображения К центру текста;
          - для правой зоны подбираем угол, ближайший к этой линии;
          - для левой зоны цель делаем зеркальной относительно оси Y
            (по сути, отражаем целевой угол: target = -dir_axis).

        Для вертикальных зон ('vertical'):
          - подстраиваемся под dominant_orientation.

        Всегда возвращаем угол в диапазоне [-90°, 90°],
        чтобы текст не был перевёрнут.
        """
        import numpy as np

        if not candidate_angles:
            return 0.0

        def norm180(a: float) -> float:
            """Нормализация угла в интервал (-180, 180]."""
            return (a + 180.0) % 360.0 - 180.0

        def clamp_to_readable(a: float) -> float:
            """
            Приводит произвольный угол к диапазону [-90, 90],
            интерпретируя его как ориентацию текста без переворота.
            """
            a = norm180(a)
            if a > 90.0:
                a -= 180.0
            elif a < -90.0:
                a += 180.0
            return a

        def pick_nearest(target_axis: float) -> float:
            """
            Выбирает из candidate_angles тот, который после
            clamp_to_readable даёт минимальное |угол - target_axis|.
            """
            best_angle = None
            best_diff = None
            for angle in candidate_angles:
                a_norm = clamp_to_readable(angle)
                diff = abs(a_norm - target_axis)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_angle = a_norm
            return 0.0 if best_angle is None else float(best_angle)

        # === БОКОВЫЕ СЕГМЕНТЫ (левая/правая зона) ===
        if zone_type == "horizontal" and region_coords is not None and img_shape is not None \
           and text_height > 0 and text_width > 0:

            H, W = img_shape

            # Центр будущего текста
            text_center_y, text_center_x = self.get_text_center_position(
                region_coords, text_height, text_width
            )

            # Угол линии ОТ центра изображения К центру текста, [0, 360)
            direction_deg = self.get_direction_to_center_for_text(
                text_center_y, text_center_x, img_shape
            )

            # Ось этой линии как ориентация в [-90, 90]
            dir_axis = norm180(direction_deg)
            if dir_axis > 90.0:
                dir_axis -= 180.0
            elif dir_axis < -90.0:
                dir_axis += 180.0

            # Определяем, в левой или правой части кадра центр текста
            img_center_x = W // 2
            is_left = text_center_x < img_center_x

            if is_left:
                # Левая зона: угол выбирается симметричным относительно оси Y,
                # т.е. целевая ось зеркальна по знаку.
                target_axis = -dir_axis
            else:
                # Правая зона: просто стремимся к оси линии центр->текст
                target_axis = dir_axis

            return pick_nearest(target_axis)

        # === ВЕРТИКАЛЬНЫЕ ЗОНЫ (верх/низ) ИЛИ ФОЛЛБЭК ===
        if dominant_orientation is not None:
            dom_axis = clamp_to_readable(float(dominant_orientation))
        else:
            dom_axis = 0.0

        return pick_nearest(dom_axis)


        
    def get_text_placement_zone(self, region_coords, img_shape):
        """
        Определяет зону размещения текста относительно центра изображения.

        Угол считаем в "математической" системе:
        0° — вправо, 90° — вверх.
        Сектор 45–135° (верхний) считаем запрещённым.
        """
        import numpy as np

        H, W = img_shape
        y_min, x_min, h, w = region_coords

        # центр региона
        y_center = y_min + h // 2
        x_center = x_min + w // 2

        # центр изображения
        img_center_y = H // 2
        img_center_x = W // 2

        # вправо +, вверх +
        dx = float(x_center - img_center_x)
        dy_math = float(img_center_y - y_center)  # инвертируем ось Y

        angle_rad = np.arctan2(dy_math, dx)
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0.0:
            angle_deg += 360.0

        # 🔴 ВЕРХНИЙ СЕКТОР 45–135° — ПОЛНОСТЬЮ ЗАПРЕЩАЕМ
        if 45.0 <= angle_deg <= 135.0:
            return "forbidden"

        # Нижний сектор 225–315° — считаем вертикальной зоной
        if 225.0 <= angle_deg <= 315.0:
            return "vertical"

        # Остальное — лево/право (горизонтальные зоны)
        return "horizontal"




    def select_region_for_text(self, txt_str, font, f_layout, f_asp,
                           place_masks, regions, *,
                           gap_px=6, min_font_px=14, shrink_step=0.90,
                           side_margin=0.86, topK=4,
                           coarse_scale=0.4, coarse_step=4, fine_step=2,
                           min_char_px_img=8, fast_mode=False, img=None,
                           nline=None, nchar=None):
        """
        Упрощённый выбор региона и размера шрифта.

        Логика:
        1) Строим дискретный набор размеров шрифта между min_font_px и f_layout
        (≈5 шагов, с минимальным шагом 1 px).
        2) Для КАЖДОГО региона (в порядке убывания площади regions['area']):
        - смотрим на bbox свободной части (mask == 0),
        - проверяем, влезает ли текст при каком-то размере из набора,
        - запоминаем КРУПНЕЙШИЙ подходящий размер шрифта для этого региона.
        3) Оставляем только регионы, где текст помещается.
        4) Сортируем подходящие регионы по площади исходного сегмента (regions['area']) по убыванию.
        5) Берём topK самых больших и случайно выбираем один из них.
        6) Возвращаем (ireg, f_fit, selected_angle).
        """
        import numpy as np
        import random

        def _dbg(msg, **kw):
            s = f"[REGION_SELECT] {msg}"
            if kw:
                extra = ", ".join(f"{k}={v}" for k, v in kw.items())
                s += " | " + extra
            print(s)

        if not place_masks:
            _dbg("no place_masks -> None")
            return None, None, None

        # --- нормализуем nline / nchar ---
        if nline is None or int(nline) <= 0:
            nline_eff = 1
        else:
            nline_eff = int(nline)

        if nchar is None or int(nchar) <= 0:
            # по умолчанию ориентируемся на длину текста
            nchar_eff = max(len(txt_str), 8)
        else:
            nchar_eff = int(nchar)

        _dbg("layout params", nline=nline_eff, nchar=nchar_eff)

        # --- дискретные шаги размера шрифта (шаг ≈ (f_max - f_min)/5) ---
        f_max = float(max(f_layout, min_font_px))
        f_min = float(min_font_px)

        if f_max < f_min:
            f_max, f_min = f_min, f_max

        if f_max == f_min:
            steps = [f_max]
        else:
            raw_step = (f_max - f_min) / 5.0
            step = max(1.0, raw_step)

            steps = []
            cur = f_max
            while cur >= f_min - 0.5:
                steps.append(cur)
                cur -= step

            # уникальные значения, отсортированные по убыванию
            steps = [max(f_min, s) for s in steps]
            steps = sorted(set(steps), reverse=True)

        _dbg("font steps", steps=[round(float(s), 1) for s in steps])

        # --- площади регионов: сначала пробуем взять из regions['area'] ---
        areas = None
        if isinstance(regions, dict) and "area" in regions:
            try:
                areas = np.asarray(regions["area"], dtype=np.float32)
                if areas.shape[0] != len(place_masks):
                    _dbg("regions['area'] length mismatch, fallback to free_area",
                        len_areas=int(areas.shape[0]), n_masks=len(place_masks))
                    areas = None
            except Exception as e:
                _dbg("failed to use regions['area']", error=repr(e))
                areas = None

        if areas is None:
            # fallback: считаем "площадь" как площадь свободной части маски
            areas = np.zeros(len(place_masks), dtype=np.float32)
            for i, pm in enumerate(place_masks):
                if pm is None:
                    continue
                arr = np.asarray(pm)
                if arr.ndim != 2:
                    continue
                free = (arr == 0)
                areas[i] = float(free.sum())
            _dbg("areas from free_area (fallback)")
        else:
            _dbg("areas from regions['area']",
                n=int(areas.size),
                max_area=float(areas.max()) if areas.size else 0.0)

        # индексы регионов по убыванию площади исходного сегмента
        order = np.argsort(-areas)

        candidate_regions = []

        for ireg in order:
            if ireg < 0 or ireg >= len(place_masks):
                continue

            mask = place_masks[ireg]
            if mask is None:
                continue

            arr = np.asarray(mask)
            if arr.ndim != 2:
                continue

            # 0 = можно, 255 = нельзя
            free = (arr == 0)
            free_area = int(free.sum())
            if free_area < (min_font_px * min_font_px * 1.5):
                # слишком маленький "рабочий" регион, пропускаем
                continue

            ys, xs = np.where(free)
            if xs.size == 0 or ys.size == 0:
                continue

            x0 = int(xs.min())
            x1 = int(xs.max())
            y0 = int(ys.min())
            y1 = int(ys.max())
            W_reg = x1 - x0 + 1
            H_reg = y1 - y0 + 1

            # минимальные размеры региона относительно пикселей текста
            if W_reg < min_char_px_img * 2 or H_reg < min_char_px_img * 1.5:
                continue

            best_f_for_region = None

            for f in steps:
                char_h = float(f)
                char_w = float(f) * float(f_asp)

                # грубая оценка размеров блока текста
                total_h = nline_eff * char_h * 1.3 + 2 * gap_px
                total_w = nchar_eff * char_w * 1.1 + 2 * gap_px

                if total_h < min_char_px_img or total_w < min_char_px_img:
                    continue

                if (total_w <= W_reg * side_margin and
                        total_h <= H_reg * side_margin):
                    # нашли КРУПНЕЙШИЙ подходящий размер для этого региона
                    best_f_for_region = float(f)
                    break

            if best_f_for_region is None:
                # в этот регион текст не влез при разумных размерах
                continue

            candidate_regions.append(
                dict(
                    ireg=int(ireg),
                    region_area=float(areas[ireg]),
                    free_area=free_area,
                    f_fit=best_f_for_region,
                    bbox=(x0, y0, x1, y1),
                )
            )

        if not candidate_regions:
            _dbg("no suitable regions after font/mask checks")
            return None, None, None

        # --- сортируем по площади исходного региона (после filter_depth/небо) ---
        candidate_regions.sort(key=lambda r: r["region_area"], reverse=True)

        if topK is None:
            topK = 4
        topK_eff = max(1, min(int(topK), len(candidate_regions)))
        top_regions = candidate_regions[:topK_eff]

        chosen = random.choice(top_regions)
        ireg_chosen = chosen["ireg"]
        f_fit = chosen["f_fit"]

        # === НОВОЕ: вычисляем угол региона и добавляем небольшой джиттер ===
        try:
            base_angle = self.estimate_region_axis_angle(place_masks[ireg_chosen])
        except Exception:
            base_angle = 0.0

        # случайное смещение вокруг оси региона, чтобы не было идеально ровно
        jitter = random.uniform(-10.0, 10.0)
        angle_raw = base_angle + jitter

        # нормализуем к "читаемому" диапазону [-90, 90], чтобы текст не был вверх ногами
        angle_norm = (angle_raw + 180.0) % 360.0 - 180.0
        if angle_norm > 90.0:
            angle_norm -= 180.0
        elif angle_norm < -90.0:
            angle_norm += 180.0

        selected_angle = float(angle_norm)

        _dbg(
            "chosen region",
            ireg=ireg_chosen,
            f_fit=round(float(f_fit), 2),
            region_area=round(float(chosen["region_area"]), 1),
            free_area=int(chosen["free_area"]),
            topK_len=len(top_regions),
            base_angle=round(float(base_angle), 1),
            selected_angle=round(selected_angle, 1),
        )

        return int(ireg_chosen), float(f_fit), selected_angle





    
    def estimate_region_axis_angle(self, region_mask):
        """
        Оценивает доминирующее направление длинной стороны свободной области региона.
        Возвращает угол в градусах относительно горизонтали в диапазоне [0, 360).
        """
        import numpy as np, cv2

        # 0 = можно, 255 = нельзя
        ys, xs = np.where(region_mask == 0)
        if ys.size < 10:
            return 0.0

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)  # ((cx, cy), (w, h), theta)
        (_, _), (w, h), theta = rect

        # ориентируемся вдоль длинной стороны
        if w < h:
            theta += 90.0

        if theta < 0.0:
            theta += 360.0
        return float(theta)


    def find_best_region_for_text(self, txt_str, lo, hi, f_asp, place_masks, cand, regions,
                                  coarse_scale, coarse_step, fine_step, side_margin, fast_mode,
                                  shrink_step, min_char_px_img, img=None):
        """
        Поиск лучшего региона для размещения текста по углам.
        """
        import numpy as np

        best_pair = (None, None)  # (ireg, f_best)
        f_probe = hi
        best_valid_angles = []
        best_ireg = None
        best_region_coords = None

        img_shape = None
        if img is not None:
            H_img, W_img = img.shape[:2]
            img_shape = (H_img, W_img)

        while f_probe >= lo:
            need_h, need_w = self.calculate_text_box_size(txt_str, f_probe, f_asp)
            need_h, need_w = self.apply_padding(need_h, need_w)

            for ireg in cand:
                Hh, Hw = place_masks[ireg].shape[:2]
                max_h = int(side_margin * Hh)
                max_w = int(side_margin * Hw)

                region_coords = self.get_region_coordinates(place_masks[ireg])
                if region_coords is None:
                    continue

                dominant_orientation = 0.0
                if img is not None:
                    try:
                        dominant_orientation = self.get_dominant_text_orientation(
                            img, place_masks[ireg]
                        )
                    except Exception:
                        dominant_orientation = 0.0

                valid_angles = []
                ok = self.check_region_fit(
                    f_probe,
                    ireg,
                    place_masks,
                    regions,
                    need_h,
                    need_w,
                    max_h,
                    max_w,
                    coarse_scale,
                    coarse_step,
                    fine_step,
                    fast_mode,
                    valid_angles,
                    min_char_px_img,
                    dominant_orientation,
                )

                if ok:
                    best_pair = (ireg, float(f_probe))
                    best_valid_angles = valid_angles
                    best_ireg = ireg
                    best_region_coords = region_coords
                    break  # нашли регион на этом f_probe

            if best_pair[0] is not None:
                break

            old_f = f_probe
            f_probe = int(f_probe * shrink_step)
            if f_probe < lo:
                return None, [], None, None

        return best_pair, best_valid_angles, best_ireg, best_region_coords



    def calculate_text_box_size(self, txt_str, font_size, aspect_ratio):
        """
        Рассчитывает размеры текстового блока для заданного текста, размера шрифта и соотношения сторон шрифта.

        txt_str: строка текста.
        font_size: размер шрифта.
        aspect_ratio: соотношение сторон шрифта (ширина к высоте).

        Возвращает:
            need_h, need_w: высота и ширина текстового блока.
        """
        # Реализация должна быть более точной, но для примера:
        # Учитываем переносы строк
        lines = txt_str.split('\n')
        max_line_width = max(len(line) for line in lines)
        
        # Ширина текста пропорциональна количеству символов в самой длинной строке
        text_width = max_line_width * font_size * aspect_ratio * 0.6  # 0.6 - коэффициент для учета реальной ширины символов
        
        # Высота текста с учетом количества строк
        line_height = font_size * 1.2  # 1.2 - межстрочный интервал
        text_height = len(lines) * line_height
        
        return int(text_height), int(text_width)

    def check_region_fit(self, f_probe, ireg, place_masks, regions, need_h, need_w,
                         max_h, max_w, coarse_scale, coarse_step, fine_step, fast_mode,
                         valid_angles, min_char_px_img, dominant_orientation=None):
        """
        Проверка, подходит ли регион для размещения текста с учетом шрифта и углов.
        Теперь углы подбираются вокруг доминирующей оси региона.
        """
        import numpy as np

        try:
            Hinv_reg = regions["homography_inv"][ireg]
            pm = place_masks[ireg]
            free_mask_fp = (pm == 0).astype(np.uint8)
            s_loc = estimate_local_scale_grid(Hinv_reg, free_mask_fp, k=9, delta=6)
            if s_loc * f_probe < min_char_px_img:
                return False
        except Exception:
            # Если что-то пошло не так с локальным масштабом — просто не используем этот фильтр
            pass

        if need_h > max_h or need_w > max_w:
            return False

        # базовый угол по геометрии региона
        try:
            base_angle = self.estimate_region_axis_angle(place_masks[ireg])
        except Exception:
            base_angle = 0.0

        check_angles = [base_angle + d for d in (-8.0, -4.0, 0.0, 4.0, 8.0)]
        check_angles = [a % 360.0 for a in check_angles]

        if dominant_orientation is not None:
            check_angles.sort(
                key=lambda x: abs((x - dominant_orientation + 180.0) % 360.0 - 180.0)
            )

        found_valid = False
        for angle in check_angles:
            ok = self.fits_text_in_region_with_angle(
                ireg,
                need_h,
                need_w,
                angle,
                place_masks,
                fast_mode,
                coarse_scale,
                coarse_step,
                fine_step,
            )
            if ok:
                valid_angles.append(angle)
                found_valid = True

        return found_valid


    def fits_text_in_region_with_angle(self, ireg, need_h, need_w, angle,
                                   place_masks, fast_mode, coarse_scale,
                                   coarse_step, fine_step):
        """
        Проверка, помещается ли текст в регион с учётом угла.
        Вращаем маску (0=можно, 255=нельзя) и проверяем через fits_coarse_to_fine.
        """
        import numpy as np

        mask = place_masks[ireg]
        Hm, Wm = mask.shape[:2]
        # Быстрый выход: текстовый блок явно шире/выше региона
        if need_h > Hm or need_w > Wm:
            return False

        rotated_forbidden = self.rotate_region_mask(mask, angle)

        ok = fits_coarse_to_fine(
            rotated_forbidden,
            need_h,
            need_w,
            coarse_scale=(0.33 if fast_mode else coarse_scale),
            coarse_step=(8 if fast_mode else coarse_step),
            fine_step=(3 if fast_mode else fine_step),
        )
        return bool(ok)


    def rotate_region_mask(self, mask, angle):
        """
        Поворачивает маску региона на указанный угол.
        Ожидается, что 0 - можно, 255 - нельзя.
        Вне исходной маски после поворота тоже считаем "нельзя".
        """
        import cv2

        h, w = mask.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_mask = cv2.warpAffine(
            mask,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,  # все снаружи считаем запрещенным
        )
        return rotated_mask
    
    
    def place_text_textfirst(self, img, place_masks, regions, gap=6,
                             min_font_px=14, start_font_px=None, start_font_px_range=None,
                             shrink_step=0.90, depth=None, occupied_global=None):
        """
        Версия с размещением текста СТРОГО внутри выбранного сегмента.
        Возвращает (img_new, text, charBB, warped_mask).
        """
        import numpy as np

        # --- helper для логов ---
        def _dbg(msg, **kw):
            s = f"[TXT] place_text_textfirst: {msg}"
            if kw:
                extra = ", ".join(f"{k}={v}" for k, v in kw.items())
                s += " | " + extra
            print(s)

        H_img, W_img = img.shape[:2]
        n_masks = len(place_masks) if place_masks is not None else 0

        _dbg(
            "start",
            img_shape=img.shape,
            n_place_masks=n_masks,
            gap=gap,
            min_font_px=min_font_px,
            start_font_px=start_font_px,
            start_font_px_range=start_font_px_range,
            shrink_step=shrink_step
        )

        if not place_masks:
            _dbg("return None: no place_masks (empty or None)")
            return None

        if 'segmentation' not in regions:
            regions['segmentation'] = place_masks
            _dbg("regions['segmentation'] not found, set from place_masks")

        # --- шрифт ---
        try:
            fs = self.text_renderer.font_state.sample()
            font = self.text_renderer.font_state.init_font(fs)
            f_asp = self.text_renderer.font_state.get_aspect_ratio(font)
        except Exception as e:
            _dbg("return None: font init failed", error=repr(e))
            return None

        short_side = float(min(H_img, W_img))
        base_char = max(min_font_px, short_side / 14.0)

        # выбор стартового размера шрифта
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

            jitter = np.random.uniform(0.9, 1.1)
            f_start_before_clip = f_start * jitter
            f_start = float(np.clip(f_start_before_clip, min_font_px, short_side / 16.0))

            f_layout = int(round(f_start * 1.30))
            if f_layout < min_font_px:
                f_layout = int(min_font_px)

            _dbg(
                "font sizing",
                source=src,
                base_char=round(base_char, 2),
                f_start_raw=round(f_start_before_clip, 2),
                f_start_clipped=round(f_start, 2),
                f_layout=f_layout,
                f_asp=round(float(f_asp), 3)
            )
        except Exception as e:
            _dbg("return None: font sizing failed", error=repr(e))
            return None

        # --- оценка (nline, nchar) --- 
        try:
            nline_raw, nchar_raw = self.text_renderer.get_nline_nchar(
                (128, 512),
                f_layout,
                f_layout * f_asp
            )
            _dbg(
                "get_nline_nchar",
                nline_raw=nline_raw,
                nchar_raw=nchar_raw
            )
        except Exception as e:
            _dbg("get_nline_nchar failed, fallback to (1, 12)", error=repr(e))
            nline_raw, nchar_raw = 1, 12

        # форсим разумные значения
        nline_eff = max(1, int(nline_raw or 1))
        if nchar_raw is None or int(nchar_raw) < 6:
            nchar_eff = 10
        else:
            nchar_eff = int(nchar_raw)

        _dbg(
            "layout target",
            nline_eff=nline_eff,
            nchar_eff=nchar_eff
        )

        # --- надёжный выбор текста ---
        try:
            txt_str = self._sample_layout_text(nline_eff, nchar_eff, max_retries=5)
        except Exception as e:
            _dbg("return None: _sample_layout_text raised", error=repr(e))
            return None

        if not txt_str or not isinstance(txt_str, str) or not txt_str.strip():
            _dbg("return None: txt_str empty or not str",
                 txt_type=str(type(txt_str)))
            return None

        txt_str = txt_str.strip()
        _dbg("txt_str chosen",
             txt_preview=(txt_str[:50] + "..." if len(txt_str) > 50 else txt_str),
             len=len(txt_str))

        # --- кэш регионов для get_region_coordinates ---
        self._regions_last = regions
        try:
            self._region_id_to_index = {id(pm): idx for idx, pm in enumerate(place_masks)}
        except Exception:
            self._region_id_to_index = {}

        # --- выбираем регион и итоговый размер ---
        try:
            ireg, f_fit, selected_angle = self.select_region_for_text(
                txt_str, font, f_layout, f_asp, place_masks, regions,
                gap_px=gap,
                min_font_px=min_font_px,
                shrink_step=shrink_step,
                side_margin=0.92,
                topK=(2 if getattr(self, "fast_mode", False) else 4),
                coarse_scale=(0.33 if getattr(self, "fast_mode", False) else 0.4),
                coarse_step=(8 if getattr(self, "fast_mode", False) else 4),
                fine_step=(3 if getattr(self, "fast_mode", False) else 2),
                min_char_px_img=getattr(self, "min_char_px_img", 8),
                fast_mode=getattr(self, "fast_mode", False),
                img=img,
                nline=nline_eff,
                nchar=nchar_eff,
            )
        except Exception as e:
            _dbg("return None: select_region_for_text raised", error=repr(e))
            return None

        if ireg is None:
            _dbg("return None: select_region_for_text returned ireg=None")
            return None

        _dbg(
            "region selected",
            ireg=int(ireg),
            f_fit=round(float(f_fit), 2),
            angle=round(float(selected_angle), 2)
        )

        key = (int(ireg), int(round(f_fit)))
        if hasattr(self, "_failed_pairs") and (key in self._failed_pairs):
            _dbg("return None: (ireg, f_fit) in _failed_pairs",
                 ireg=int(ireg),
                 f_fit=int(round(f_fit)))
            return None

        # --- проверка локального масштаба ---
        try:
            Hinv_reg = regions['homography_inv'][ireg]
            region_mask = place_masks[ireg]
            free_mask_fp = (region_mask == 0).astype(np.uint8)

            s_loc = estimate_local_scale_grid(Hinv_reg, free_mask_fp, k=9, delta=6)
            _dbg(
                "local scale",
                s_loc=round(float(s_loc), 3),
                min_char_px_img=getattr(self, "min_char_px_img", 8),
                s_loc_times_f_fit=round(float(s_loc * f_fit), 2)
            )

            if s_loc * f_fit < getattr(self, "min_char_px_img", 8):
                _dbg("return None: s_loc * f_fit too small",
                     s_loc_times_f_fit=round(float(s_loc * f_fit), 2))
                return None
        except Exception as e:
            _dbg("local scale check failed (ignored)", error=repr(e))

        # --- финальный размер шрифта ---
        try:
            font.size = self.text_renderer.font_state.get_font_size(font, f_fit)
            _dbg("font.size set", f_fit=round(float(f_fit), 2), font_size=font.size)
        except Exception as e:
            _dbg("return None: get_font_size failed", error=repr(e))
            return None

        # --- координаты региона ---
        region_mask = place_masks[ireg]
        try:
            region_coords = self.get_region_coordinates(region_mask)
        except Exception as e:
            _dbg("return None: get_region_coordinates raised", error=repr(e))
            return None

        if region_coords is None:
            _dbg("return None: get_region_coordinates returned None")
            return None

        _dbg(
            "region_coords ok",
            n_points=(len(region_coords) if hasattr(region_coords, "__len__") else "?"
                      )
        )

        # --- рендер текста в регион ---
        try:
            img_new, bb_img, text_mask_img = self.render_text_overlay(
                img, txt_str, font,
                selected_angle=selected_angle,
                region_coords=region_coords,
                depth=depth
            )
        except Exception as e:
            _dbg("return None: render_text_overlay raised", error=repr(e))
            return None

        if img_new is None:
            _dbg("return None: img_new is None after render_text_overlay")
            return None

        if text_mask_img is None:
            _dbg("return None: text_mask_img is None after render_text_overlay")
            return None

        # немного статистики по маске, чтобы видеть, что реально нарисовали
        try:
            mask_area = int((text_mask_img > 0).sum())
            _dbg(
                "render result",
                mask_area=mask_area,
                bbox_is_none=(bb_img is None),
                H_img=H_img,
                W_img=W_img
            )
            if mask_area == 0:
                _dbg("return None: text_mask_img empty (area=0)")
                return None
        except Exception as e:
            _dbg("mask stats failed (ignored)", error=repr(e))

        # запоминаем последний удачный размер
        self.last_font_h_px = f_fit
        _dbg("SUCCESS", txt_len=len(txt_str), ireg=int(ireg))

        return img_new, txt_str, bb_img, text_mask_img


    def get_region_coordinates(self, region_mask):
        """
        Возвращает координаты региона в КООРДИНАТАХ ИЗОБРАЖЕНИЯ
        в виде 4 углов [[x0,y0],[x1,y0],[x1,y1],[x0,y1]].

        ВАЖНО:
        - place_mask: 0 = можно, 255 = нельзя → берём bbox только по "нулям".
        - сами маски хранятся во фронто-параллельных координатах, поэтому
          при наличии homography_inv проецируем bbox в image-space.
        """
        import numpy as np

        # 1) bbox по "разрешённой" области (0 = можно)
        ys, xs = np.where(region_mask == 0)
        if xs.size == 0:
            print("[REGION] empty free area in region_mask -> None")
            return None

        x0_fp, x1_fp = xs.min(), xs.max()
        y0_fp, y1_fp = ys.min(), ys.max()

        corners_fp = np.array([
            [x0_fp, y0_fp],
            [x1_fp, y0_fp],
            [x1_fp, y1_fp],
            [x0_fp, y1_fp],
        ], dtype=np.float32)

        # 2) пробуем спроецировать в координаты исходного изображения
        Hinv = None
        ireg = None
        regions = getattr(self, "_regions_last", None)
        id2idx = getattr(self, "_region_id_to_index", None)

        if regions is not None and id2idx is not None:
            ireg = id2idx.get(id(region_mask), None)
            try:
                if ireg is not None and "homography_inv" in regions:
                    Hinvs = regions["homography_inv"]
                    if ireg < len(Hinvs):
                        Hinv = Hinvs[ireg]
            except Exception as e:
                print("[REGION] failed to fetch Hinv for region:", repr(e))
                Hinv = None

        if Hinv is not None:
            try:
                # _warp_points: (Hinv, pts_xy[N,2]) -> (N,2) в image-space
                corners_img = _warp_points(Hinv, corners_fp)
                coords = corners_img.astype(np.float32)
                print(
                    f"[REGION] ireg={ireg}, "
                    f"bbox_fp=({x0_fp},{y0_fp})-({x1_fp},{y1_fp}), "
                    f"corners_img={coords.tolist()}"
                )
                return coords
            except Exception as e:
                print("[REGION] _warp_points failed, fallback to FP coords:", repr(e))

        # fallback: отдаём bbox в координатах маски (FP)
        print(
            f"[REGION] no Hinv, using FP bbox: "
            f"({x0_fp},{y0_fp})-({x1_fp},{y1_fp})"
        )
        return corners_fp

    
    def get_region_center_img(self, ireg, place_masks, regions, img_shape):
        """
        Центр региона (y, x) в координатах изображения.

        - При no_geom=True маска уже в image-space → берём центр по 0-пикселям.
        - При no_geom=False маска в фронто-параллельной системе:
          считаем центр там и проектируем через Hinv в image-space.
        """
        import numpy as np

        H_img, W_img = img_shape
        mask_fp = place_masks[ireg]

        ys, xs = np.where(mask_fp == 0)
        if ys.size == 0:
            return H_img // 2, W_img // 2

        cy_fp = float(ys.mean())
        cx_fp = float(xs.mean())

        use_geom = (not getattr(self, "no_geom", False)) \
                   and ("homography_inv" in regions) \
                   and (ireg < len(regions["homography_inv"]))

        if use_geom:
            Hinv = regions["homography_inv"][ireg]
            pts  = np.array([[cx_fp, cy_fp]], dtype=np.float32)  # (1,2)
            w    = _warp_points(Hinv, pts)  # (1,2) в image-space
            cx_img = float(w[0, 0])
            cy_img = float(w[0, 1])
        else:
            cx_img, cy_img = cx_fp, cy_fp

        cy_img = int(max(0, min(H_img - 1, round(cy_img))))
        cx_img = int(max(0, min(W_img - 1, round(cx_img))))
        return cy_img, cx_img


    def apply_padding(self, need_h, need_w, pad_rel=0.06):
        """
        Применяет дополнительный зазор вокруг текста (паддинг).

        need_h: исходная высота текстового блока.
        need_w: исходная ширина текстового блока.
        pad_rel: коэффициент добавления зазора (например, 6%).
        
        Возвращает:
            new_h, new_w: высота и ширина текстового блока с добавленным зазором.
        """
        new_h = int(need_h * (1 + pad_rel))  # Увеличиваем высоту на 6%
        new_w = int(need_w * (1 + pad_rel))  # Увеличиваем ширину на 6%
        return new_h, new_w
    
    def _boost_text_neon(self, img, text_mask):
        """
        Локально усиливает яркость и насыщенность текста под маской,
        чтобы он выглядел как яркая вывеска.
        """
        import numpy as np
        import cv2

        if img is None or text_mask is None:
            return img
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        H, W = img.shape[:2]
        if text_mask.shape[:2] != (H, W):
            # на всякий случай подгоним маску
            text_mask = cv2.resize(text_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        mask = (text_mask > 0)
        if not np.any(mask):
            return img

        # Работаем в HSV, чтобы аккуратно поднять насыщенность и яркость
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        s = s.astype(np.float32)
        v = v.astype(np.float32)

        # Усиливаем насыщенность и яркость ТОЛЬКО под текстом
        s[mask] = np.clip(s[mask] * 1.45 + 25.0, 0.0, 255.0)
        v[mask] = np.clip(v[mask] * 1.55 + 35.0, 0.0, 255.0)

        s = s.astype(np.uint8)
        v = v.astype(np.uint8)
        hsv_boost = cv2.merge([h, s, v])
        img_boost = cv2.cvtColor(hsv_boost, cv2.COLOR_HSV2RGB)

        return img_boost


    def _apply_partial_text_occlusion(self, img, text_mask, bg_img, max_frac: float = 0.20):
        """
        Частично перекрывает текст мягкими, плавными «пятнами».
        text_mask : uint8 0/255, где 255 = текст
        bg_img    : исходное RGB-изображение без текста (обычно rgb из входа render_text)
        max_frac  : максимум доля пикселей текста, которую можно перекрыть (0..1)
        """
        import numpy as np
        import cv2

        if img is None or text_mask is None:
            return img

        H, W = img.shape[:2]

        # Приводим img к 3-канальному виду для удобства композитинга
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = img

        # Фон: либо исходный rgb, либо текущий img
        if bg_img is None or bg_img.shape[:2] != (H, W):
            bg = img_rgb
        else:
            bg = bg_img

        # Бинарная маска текста
        text_bin = (text_mask > 0).astype(np.uint8)
        total = int(text_bin.sum())
        if total <= 0:
            return img

        # Целевая доля перекрытия (от 5% до max_frac, но не более max_frac)
        frac = float(np.random.uniform(0.15, max_frac))
        frac = min(max(frac, 0.0), max_frac)
        target = int(total * frac)
        if target < 20:
            # слишком мало — визуально почти не видно, пропустим
            return img

        ys, xs = np.where(text_bin > 0)
        if ys.size == 0:
            return img

        occ_bin = np.zeros_like(text_bin, dtype=np.uint8)
        occluded = 0
        max_iters = 24

        # Строим несколько «пятен» как эллипсы внутри текста
        for _ in range(max_iters):
            if occluded >= target:
                break

            idx = np.random.randint(0, ys.size)
            cy, cx = int(ys[idx]), int(xs[idx])

            # базовый размер пятна ~ sqrt(целевой площади), но с разбросом
            base = max(6, int(np.sqrt(target) * np.random.uniform(0.25, 0.6)))
            ry = int(base * np.random.uniform(0.6, 1.4))
            rx = int(base * np.random.uniform(0.6, 1.4))

            y0 = max(0, cy - ry)
            y1 = min(H, cy + ry)
            x0 = max(0, cx - rx)
            x1 = min(W, cx + rx)
            if y1 <= y0 or x1 <= x0:
                continue

            patch = np.zeros_like(text_bin, dtype=np.uint8)
            center = (cx, cy)
            axes = (max(1, rx), max(1, ry))
            angle = float(np.random.uniform(0.0, 180.0))
            cv2.ellipse(patch, center, axes, angle, 0.0, 360.0, 1, -1)

            # Ограничиваемся текстом и ещё не перекрытыми пикселями
            patch = patch & text_bin & (occ_bin == 0)
            added = int(patch.sum())
            if added == 0:
                continue

            # Если этот патч перекрывает слишком много — обрезаем до остатка
            if occluded + added > target:
                need = target - occluded
                if need <= 0:
                    break
                ys_r, xs_r = np.where(patch > 0)
                if ys_r.size > need:
                    sel = np.random.choice(ys_r.size, size=need, replace=False)
                    patch2 = np.zeros_like(patch, dtype=np.uint8)
                    patch2[ys_r[sel], xs_r[sel]] = 1
                    patch = patch2
                    added = need

            occ_bin |= patch
            occluded += added

        if occluded == 0:
            return img

        # Превращаем жёсткую маску в мягкую: размываем Гауссом
        k = int(max(5, (H + W) * 0.01))
        if k % 2 == 0:
            k += 1
        occ_soft = cv2.GaussianBlur(occ_bin.astype(np.float32), (k, k), 0)

        if occ_soft.max() > 0:
            occ_soft /= occ_soft.max()
        occ_soft = np.clip(occ_soft, 0.0, 1.0)

        # Итоговая "прозрачность" перекрытия
        alpha = float(np.random.uniform(0.7, 1.0))
        occ_alpha = (occ_soft * alpha).astype(np.float32)[..., None]

        img_f = img_rgb.astype(np.float32)
        bg_f = bg.astype(np.float32)

        # Мягкий композит: текст местами «прячется» под фоном/объектами
        out = img_f * (1.0 - occ_alpha) + bg_f * occ_alpha
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)

        # Возвращаем в том же формате, что и пришло
        if img.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        return out

    def _sample_layout_text(self, nline, nchar, max_retries=5):
        """
        Надёжный выбор ТОЛЬКО ОДНОГО СЛОВА под заданные (nline, nchar).

        Логика:
        - несколько попыток text_source.sample(...)
        - приводим результат к строке
        - из строки выбираем одно слово длиной >= self.min_word_len
        - если всё плохо — отдаём fallback-слово.
        """
        import random

        min_len = int(getattr(self, "min_word_len", 4))
        txt_str = None
        raw_obj = None

        for attempt in range(max_retries):
            try:
                kind = tu.sample_weighted(self.text_renderer.p_text)
            except Exception:
                kind = None

            try:
                raw_obj = self.text_renderer.text_source.sample(nline, nchar, kind)
            except Exception:
                raw_obj = None

            # Приводим к строке
            if isinstance(raw_obj, list):
                parts = [str(l).strip() for l in raw_obj if str(l).strip()]
                txt_str = " ".join(parts)
            elif isinstance(raw_obj, str):
                txt_str = raw_obj.strip()
            else:
                txt_str = None

            if not txt_str:
                continue

            # Разбиваем на слова и выбираем одно подходящее
            words = [w for w in txt_str.split() if len(w) >= min_len]
            if not words:
                continue

            # Одно слово на блок
            return random.choice(words)

        # --- Fallback — чтобы генерация вообще не умирала ---
        fallback = "пример"
        return fallback


    def render_text_overlay(self, img, txt_str, font, selected_angle, region_coords, depth=None):
        """
        Рисует txt_str внутри региона.

        - region_coords может быть:
            * массив точек (N, 2) в координатах изображения;
            * 4 числа: (x0, y0, x1, y1) / (y0, x0, y1, x1) / (x, y, w, h).

        Возвращает:
            img_new       — изображение с текстом,
            charBB        — (2, 4, N_chars) в глобальных координатах,
            text_mask_img — маска текста в глобальных координатах.
        """
        import numpy as np
        import cv2
        import pygame
        import random

        try:
            H, W = img.shape[:2]
            print(
                f"[OVERLAY] start | txt='{txt_str[:32]}' len={len(txt_str)}, "
                f"type(region_coords)={type(region_coords)}, "
                f"has_shape={hasattr(region_coords, 'shape')}, "
                f"selected_angle={selected_angle}"
            )

            # -------- 1. Нормализуем region_coords в bbox (x0, y0, x1, y1) --------
            x0 = y0 = x1 = y1 = None

            rc = np.asarray(region_coords)
            if rc.ndim == 2 and rc.shape[1] == 2:
                # Интерпретируем как полигон
                print(f"[OVERLAY] region_coords as polygon, shape={rc.shape}")
                xs = rc[:, 0]
                ys = rc[:, 1]
                x0 = float(xs.min())
                x1 = float(xs.max())
                y0 = float(ys.min())
                y1 = float(ys.max())
            else:
                # Плоский формат: 4 числа
                flat = np.asarray(region_coords, dtype=np.float32).ravel()
                print(f"[OVERLAY] region_coords flat={flat}, shape={flat.shape}")

                if flat.size != 4:
                    print("[OVERLAY] unsupported region_coords format, size != 4")
                    return None, None, None

                a, b, c, d = flat.tolist()
                candidates = []

                def _check_xyxy(x0_, y0_, x1_, y1_):
                    vals = np.array([x0_, y0_, x1_, y1_], dtype=np.float32)
                    if not np.isfinite(vals).all():
                        return None
                    if x1_ <= x0_ + 5 or y1_ <= y0_ + 5:
                        return None
                    if x0_ < -5 or y0_ < -5 or x1_ > W + 5 or y1_ > H + 5:
                        return None
                    return (x0_, y0_, x1_, y1_)

                def _check_yx_yx(y0_, x0_, y1_, x1_):
                    vals = np.array([x0_, y0_, x1_, y1_], dtype=np.float32)
                    if not np.isfinite(vals).all():
                        return None
                    if x1_ <= x0_ + 5 or y1_ <= y0_ + 5:
                        return None
                    if x0_ < -5 or y0_ < -5 or x1_ > W + 5 or y1_ > H + 5:
                        return None
                    return (x0_, y0_, x1_, y1_)

                def _check_xywh(x_, y_, w_, h_):
                    vals = np.array([x_, y_, w_, h_], dtype=np.float32)
                    if not np.isfinite(vals).all():
                        return None
                    if w_ <= 5 or h_ <= 5:
                        return None
                    x0_ = x_
                    y0_ = y_
                    x1_ = x_ + w_
                    y1_ = y_ + h_
                    if x0_ < -5 or y0_ < -5 or x1_ > W + 5 or y1_ > H + 5:
                        return None
                    return (x0_, y0_, x1_, y1_)

                cand1 = _check_xyxy(a, b, c, d)
                if cand1 is not None:
                    candidates.append(("xyxy", cand1))

                cand2 = _check_yx_yx(a, b, c, d)
                if cand2 is not None:
                    candidates.append(("yx_yx", cand2))

                cand3 = _check_xywh(a, b, c, d)
                if cand3 is not None:
                    candidates.append(("xywh", cand3))

                if not candidates:
                    print("[OVERLAY] no valid bbox interpretation for flat coords")
                    return None, None, None

                # выбираем bbox с МИНИМАЛЬНОЙ площадью
                best_type, (x0, y0, x1, y1) = min(
                    candidates,
                    key=lambda kv: (kv[1][2] - kv[1][0]) * (kv[1][3] - kv[1][1])
                )
                print(
                    f"[OVERLAY] interpreted flat as {best_type}: "
                    f"x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}"
                )

            # -------- 2. Проверка bbox и небольшая "усадка" внутрь --------
            if x0 is None or y0 is None or x1 is None or y1 is None:
                print("[OVERLAY] bbox not resolved")
                return None, None, None

            x0 = max(0.0, min(float(x0), W - 1.0))
            x1 = max(0.0, min(float(x1), W - 1.0))
            y0 = max(0.0, min(float(y0), H - 1.0))
            y1 = max(0.0, min(float(y1), H - 1.0))

            if x1 <= x0 + 5 or y1 <= y0 + 5:
                print(
                    f"[OVERLAY] bbox too small after clamp: "
                    f"x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}"
                )
                return None, None, None

            margin_x = (x1 - x0) * 0.05
            margin_y = (y1 - y0) * 0.10
            x0i = int(round(x0 + margin_x))
            x1i = int(round(x1 - margin_x))
            y0i = int(round(y0 + margin_y))
            y1i = int(round(y1 - margin_y))

            if x1i <= x0i + 4 or y1i <= y0i + 4:
                print(
                    f"[OVERLAY] bbox collapsed after margin: "
                    f"x0i={x0i}, y0i={y0i}, x1i={x1i}, y1i={y1i}"
                )
                return None, None, None

            print(
                f"[OVERLAY] final bbox for text: "
                f"x0={x0i}, y0={y0i}, x1={x1i}, y1={y1i}"
            )

            W_reg = x1i - x0i
            H_reg = y1i - y0i

            # -------- 3. Рендер текста в локальный патч через pygame --------
            try:
                pygame.init()
            except Exception:
                pass

            surf = pygame.Surface((W_reg, H_reg), flags=pygame.SRCALPHA)
            surf.fill((0, 0, 0, 0))

            try:
                font.origin = True
            except Exception:
                pass

            base_size = getattr(font, "size", None)
            if base_size is None:
                base_size = int(max(8, min(H_reg, W_reg) / 2))
                try:
                    font.size = base_size
                except Exception:
                    pass

            # уменьшаем шрифт, пока текст не влезет в локальный bbox
            for _ in range(12):
                try:
                    text_rect = font.get_rect(txt_str)
                except Exception:
                    try:
                        text_rect = font.get_rect(txt_str)[0]
                    except Exception as e:
                        print("[OVERLAY] font.get_rect failed:", repr(e))
                        return None, None, None

                if (
                    text_rect.width <= W_reg * 0.9
                    and text_rect.height <= H_reg * 0.8
                ):
                    break
                try:
                    font.size = max(4, int(font.size * 0.9))
                except Exception:
                    pass

            tx = (W_reg - text_rect.width) // 2
            ty = (H_reg - text_rect.height) // 2 + text_rect.height

            # ---- яркий цвет текста ----
            bright_colors = [
                (255, 255, 255),
                (255, 255, 0),
                (0, 255, 255),
                (255, 128, 0),
                (0, 255, 0),
                (255, 0, 255),
                (0, 128, 255),
            ]
            fg_color = random.choice(bright_colors)

            try:
                # Сначала зальём текстом (основной цвет)
                font.render_to(surf, (tx, ty), txt_str, fg_color)
            except Exception as e:
                print("[OVERLAY] font.render_to failed:", repr(e))
                return None, None, None

            text_rgb = pygame.surfarray.pixels3d(surf).copy()
            text_a = pygame.surfarray.pixels_alpha(surf).copy()
            mask_local = (text_a > 0).astype(np.uint8) * 255

            if int(mask_local.sum()) < 10:
                print("[OVERLAY] empty text mask in local bbox")
                return None, None, None

            text_bgr = cv2.cvtColor(text_rgb, cv2.COLOR_RGB2BGR)

            # -------- 3.1 Подсчёт числа символов (без пробелов) --------
            chars = [c for c in txt_str if not c.isspace()]
            n_chars = len(chars)

            # >>> 3.2 Поворот патча текста по selected_angle <<< 
            angle = 0.0
            try:
                if selected_angle is not None:
                    angle = float(selected_angle)
            except Exception:
                angle = 0.0

            if abs(angle) > 0.5:
                h_t, w_t = mask_local.shape[:2]
                center = (w_t / 2.0, h_t / 2.0)

                M = cv2.getRotationMatrix2D(center, angle, 1.0)

                text_bgr = cv2.warpAffine(
                    text_bgr,
                    M,
                    (w_t, h_t),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                mask_local = cv2.warpAffine(
                    mask_local,
                    M,
                    (w_t, h_t),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

                print(f"[OVERLAY] rotated text patch by {angle:.2f} deg")
            else:
                M = None  # на всякий случай

            # -------- 3.3 Делаем текст чуть ТОНЬШЕ (лёгкая эрозия маски) --------
            kernel_thin = np.ones((2, 2), np.uint8)
            mask_thin = cv2.erode(mask_local, kernel_thin, iterations=1)
            if mask_thin.sum() > 0:
                mask_local = mask_thin

            # -------- 4. Вклеиваем патч в исходное изображение + ЧЁРНАЯ ОБВОДКА --------
            img_new = img.copy()
            patch = img_new[y0i:y1i, x0i:x1i].copy()

            if patch.shape[:2] != text_bgr.shape[:2]:
                print("[OVERLAY] shape mismatch patch vs text, resizing text")
                text_bgr = cv2.resize(
                    text_bgr,
                    (patch.shape[1], patch.shape[0])
                )
                mask_local = cv2.resize(
                    mask_local,
                    (patch.shape[1], patch.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # ---- 4.1 Аккуратное зеркалирование текста по вертикальной оси (лево↔право) ----
            ys_m, xs_m = np.where(mask_local > 0)
            if xs_m.size > 0 and ys_m.size > 0:
                x0_m, x1_m = int(xs_m.min()), int(xs_m.max())
                y0_m, y1_m = int(ys_m.min()), int(ys_m.max())

                roi_y = slice(y0_m, y1_m + 1)
                roi_x = slice(x0_m, x1_m + 1)

                sub_text = text_bgr[roi_y, roi_x]
                sub_mask = mask_local[roi_y, roi_x]

                sub_text_flipped = cv2.flip(sub_text, 1)
                sub_mask_flipped = cv2.flip(sub_mask, 1)

                text_bgr[roi_y, roi_x] = sub_text_flipped
                mask_local[roi_y, roi_x] = sub_mask_flipped

                print(
                    f"[OVERLAY] mirrored text horizontally inside ROI "
                    f"x=[{x0_m},{x1_m}], y=[{y0_m},{y1_m}]"
                )

            # ---- строим контур: dilate(mask) - mask ----
            kernel_outline = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask_local, kernel_outline, iterations=2)
            outline_mask = cv2.subtract(dilated, mask_local)
            outline_mask = (outline_mask > 0).astype(np.uint8) * 255

            # сначала кладём контур (чёрный), потом сам текст
            m_outline = outline_mask.astype(bool)
            m_text = mask_local.astype(bool)

            # чёрный контур
            patch[m_outline] = (0, 0, 0)

            # цветной текст
            patch[m_text] = text_bgr[m_text]

            # 🔧 ВАЖНО: возвращаем обратно ровно в тот же x-срез
            img_new[y0i:y1i, x0i:x1i] = patch

            # глобальная маска (в координатах всей картинки) — только сам текст
            text_mask_img = np.zeros((H, W), dtype=np.uint8)
            text_mask_img[y0i:y1i, x0i:x1i] = mask_local

            # -------- 5. Строим charBB по глобальной маске через PCA --------
            charBB = None
            if n_chars > 0:
                ys, xs = np.where(text_mask_img > 0)
                if xs.size == 0 or ys.size == 0:
                    print("[OVERLAY] text_mask_img is empty, charBB=None")
                    return img_new, None, text_mask_img

                # точки в (x, y)
                pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
                mean = pts.mean(axis=0)  # (mx, my)
                pts_centered = pts - mean[None, :]

                # --- PCA, главная ось строки ---
                cov = np.cov(pts_centered, rowvar=False)
                evals, evecs = np.linalg.eigh(cov)
                main_idx = int(np.argmax(evals))
                d0 = evecs[:, main_idx].astype(np.float32)  # главная ось
                norm = float(np.linalg.norm(d0))
                if norm < 1e-6:
                    d0 = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    d0 /= norm

                # направляем d0 вправо по X
                if d0[0] < 0:
                    d0 = -d0

                # ортогональная ось (толщина строки)
                d1 = np.array([-d0[1], d0[0]], dtype=np.float32)

                # проекции точек на оси
                us = pts_centered @ d0
                vs = pts_centered @ d1
                umin, umax = float(us.min()), float(us.max())
                vmin, vmax = float(vs.min()), float(vs.max())

                du = umax - umin
                dv = vmax - vmin

                # небольшой запас
                umin -= 0.02 * du
                umax += 0.02 * du
                vmin -= 0.05 * dv
                vmax += 0.05 * dv

                charBB = np.zeros((2, 4, n_chars), dtype=np.float32)

                for i in range(n_chars):
                    t0 = float(i) / n_chars
                    t1 = float(i + 1) / n_chars
                    u0 = umin + t0 * (umax - umin)
                    u1 = umin + t1 * (umax - umin)

                    # 4 угла прямоугольника для i-го символа
                    p0 = mean + u0 * d0 + vmin * d1
                    p1 = mean + u1 * d0 + vmin * d1
                    p2 = mean + u1 * d0 + vmax * d1
                    p3 = mean + u0 * d0 + vmax * d1

                    quad = np.stack([p0, p1, p2, p3], axis=0)  # (4, 2)
                    charBB[0, :, i] = quad[:, 0]
                    charBB[1, :, i] = quad[:, 1]

                # обрезаем за границы изображения
                charBB[0, :, :] = np.clip(charBB[0, :, :], 0, W - 1)
                charBB[1, :, :] = np.clip(charBB[1, :, :], 0, H - 1)

                print(f"[OVERLAY] charBB built from PCA for {n_chars} chars")
            else:
                print("[OVERLAY] no non-space chars, charBB=None")

            return img_new, charBB, text_mask_img

        except Exception as e:
            import traceback
            print("[OVERLAY] UNHANDLED EXCEPTION:", repr(e))
            traceback.print_exc()
            return None, None, None



    def rotate_text_and_bbox(self, text_img, bbox, angle):
        import numpy as np
        import cv2

        h, w = text_img.shape[:2]
        
        # Вычисляем новый размер изображения после поворота без обрезки
        angle_rad = np.deg2rad(angle)
        new_w = int(abs(w * np.cos(angle_rad)) + abs(h * np.sin(angle_rad)))
        new_h = int(abs(w * np.sin(angle_rad)) + abs(h * np.cos(angle_rad)))
        
        # Центр исходного изображения
        center = (w // 2, h // 2)
        # Центр нового изображения
        new_center = (new_w // 2, new_h // 2)
        
        # Получаем матрицу поворота
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Корректируем матрицу для размещения всего изображения в новом холсте
        M[0, 2] += new_center[0] - center[0]
        M[1, 2] += new_center[1] - center[1]
        
        # Поворачиваем изображение с новыми размерами
        rotated_text_img = cv2.warpAffine(
            text_img, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Поворачиваем bounding box
        num_chars = bbox.shape[2]
        pts = bbox.reshape(2, -1)
        ones = np.ones((1, pts.shape[1]), dtype=pts.dtype)
        pts_h = np.vstack([pts, ones])
        
        rotated_pts = M @ pts_h
        rotated_bbox = rotated_pts.reshape(2, 4, num_chars)

        return rotated_text_img, rotated_bbox


    def get_min_h(self, bb, text):
        # find min-height:
        h = np.linalg.norm(bb[:,3,:] - bb[:,0,:], axis=0)
        # remove newlines and spaces:
        text = ''.join(text.split())
        assert len(text)==bb.shape[-1]

        alnum = np.array([ch.isalnum() for ch in text])
        h = h[alnum]
        return np.min(h)

    def get_num_text_regions(self, nregions: int) -> int:
        """
        Сколько текстовых блоков (здесь = СЛОВ) пытаться разместить на изображении.

        Ограничения:
        - по умолчанию 2–4 слова на картинку;
        - но не больше количества доступных регионов nregions.
        При желании можно переопределить self.min_words_per_image / self.max_words_per_image.
        """
        import numpy as np

        # Диапазон слов на изображении по умолчанию
        default_min = 2
        default_max = 4

        # Можно переопределить извне как атрибуты объекта
        min_blocks = int(getattr(self, "min_words_per_image", default_min))
        max_blocks = int(getattr(self, "max_words_per_image", default_max))

        if max_blocks < min_blocks:
            max_blocks = min_blocks

        if nregions <= 0:
            return 0

        # случайное число блоков (слов) в этом кадре
        k = int(np.random.randint(min_blocks, max_blocks + 1))

        # не больше числа регионов
        k = max(1, min(k, int(nregions)))
        return k

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
        """
        Основной генератор:
        1) строит регионы и их place_mask,
        2) несколько раз пытается разместить текст по стратегии text-first
           (ТЕПЕРЬ: один блок = одно слово),
        3) следит за глобальными коллизиями,
        4) добавляет шум / затемнение,
        5) при viz=True рисует дебаг-картинки.

        Возвращает список словарей {'img','charBB','wordBB','txt'}.
        """
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        rgb = to_rgb(rgb)

        from noise_utils import degrade_scene_rgb  # если нужно, можешь использовать

        # --- строим регионы и place_mask ---
        try:
            print("[render_text] start: rgb_shape =", rgb.shape, ", seg_shape =", seg.shape, ", ninstance =", ninstance)

            # 1) xyz из depth – как в оригинале SynthText
            xyz = su.DepthCamera.depth2xyz(depth)

            # 2) регионы по сегментации
            regions = TextRegions.get_regions(xyz, seg, area, label)
            print("[render_text] TextRegions.get_regions ->", len(regions["label"]), "raw regions")

            # 3) отфильтровали только планарные области (подбираем одну плоскость на регион)
            regions = TextRegions.filter_depth(xyz, seg, regions)
            print("[render_text] TextRegions.filter_depth ->", len(regions["label"]), "planar regions")

            # 4) строим fronto-parallel маски и гомографии, как в оригинальном коде
            regions = self.filter_for_placement(xyz, seg, regions)

            # --- важный ранний выход: НИЧЕГО НЕ НАШЛИ -> вернём ПУСТОЙ СПИСОК ---
            if regions is None or len(regions.get("place_mask", [])) == 0:
                print("[render_text] no regions after filter_for_placement, early exit")

                # опционально сохраним дебаг-картинки
                try:
                    _save_img(f"{_stamp()}_DBG_rgb.png", rgb)
                    _save_img(
                        f"{_stamp()}_DBG_seg.png",
                        (seg.astype(np.float32) / max(1, seg.max()) * 255).astype(np.uint8),
                    )
                    d = depth.astype(np.float32)
                    d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    _save_img(f"{_stamp()}_DBG_depth.png", d_norm)
                except Exception:
                    pass

                return []  # <--- ВАЖНО: список, а не (rgb, None, None)

            nregions = len(regions['place_mask'])
            if nregions < 1:
                return []

            # сколько блоков (слов) хотим в среднем
            target_blocks = self.get_num_text_regions(nregions)

            # динамические бюджеты (сколько попыток на картинку и на регион)
            global_budget, per_region_cap, max_shrink_trials = self._compute_budgets(
                nregions, target_blocks
            )
            self.global_attempt_budget = global_budget
            self.per_region_attempt_cap = per_region_cap
            self._max_shrink_trials_runtime = max_shrink_trials
        except Exception as e:
            print("[render_text] exception during region prep:", repr(e))
            return []

        res = []
        H0, W0 = rgb.shape[:2]

        for i in range(ninstance):
            img = rgb.copy()
            itext, ibb = [], []

            # глобальная карта занятости в координатах изображения
            occupied_global = np.zeros(img.shape[:2], dtype=np.uint8)

            # сколько текстовых блоков (слов) хотим в этом экземпляре
            target_blocks = self.get_num_text_regions(nregions)

            attempts_left = int(getattr(self, "global_attempt_budget", 10))
            placed_count = 0

            while attempts_left > 0 and placed_count < target_blocks:
                attempts_left -= 1

                try:
                    txt_render_res = self.place_text_textfirst(
                        img,
                        place_masks=regions['place_mask'],
                        regions=regions,
                        gap=getattr(self, "min_box_gap_rect_px", 8),
                        min_font_px=MIN_FONT_PX,
                        shrink_step=SHRINK_STEP,
                        depth=depth,
                        occupied_global=occupied_global
                    )
                    if txt_render_res is None:
                        continue
                    img_new, text, bb, warped_mask = txt_render_res
                except Exception:
                    continue

                if warped_mask is None:
                    continue

                # Проверка пересечения с уже размещёнными блоками
                m_img = (warped_mask > 0).astype(np.uint8) * 255
                overlap = cv2.bitwise_and(occupied_global, m_img)
                if int(overlap.sum()) > 0:
                    # коллизия — пропускаем
                    continue

                # Обновляем глобальную занятость с зазором между блоками
                gap_px = int(getattr(self, "min_box_gap_px", 12))
                k = 2 * gap_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                m_inflated = cv2.dilate(m_img, kernel, 1)
                occupied_global = np.maximum(occupied_global, m_inflated)

                # Фиксируем картинку и аннотации
                img = img_new
                itext.append(text)   # ОДНО СЛОВО на блок
                ibb.append(bb)
                placed_count += 1

            # лёгкие аугментации сцены (шумы / цвет)
            img = apply_random_augmentations(img)

            # если в этом инстансе вообще ничего не поставилось — пропускаем
            if placed_count == 0:
                continue

            # Собираем ответ по инстансу
            idict = {'img': img, 'txt': itext, 'charBB': None, 'wordBB': None}
            if len(ibb):
                idict['charBB'] = np.concatenate(ibb, axis=2)

            H, W = img.shape[:2]
            if idict['charBB'] is not None:
                # text = "слово1 слово2 ...", wordBB на каждое слово
                idict['wordBB'] = self.char2wordBB(
                    idict['charBB'].copy(), ' '.join(itext),
                    pad_px=4, pad_rel=0.05, clamp_shape=(H, W)
                )

            # Пост-обработка сцены (затемнение, «плохая камера»)
            try:
                from noise_utils import darken_scene_realistic
                idict['img'] = darken_scene_realistic(idict['img'])
            except Exception:
                pass

            try:
                from noise_utils import noise_bad_camera_random
                idict['img'] = noise_bad_camera_random(idict['img'])
            except Exception:
                pass

            # Визуализация (если включено)
            if viz:
                try:
                    if idict['wordBB'] is not None:
                        viz_textbb(1, idict['img'], [idict['wordBB']], alpha=1.0)
                    else:
                        import matplotlib.pyplot as plt
                        plt.figure(1)
                        plt.clf()
                        plt.imshow(_rgb(idict['img']))
                        plt.axis('off')
                        if not _plt_stable_draw(plt.gcf(), pause=0.35):
                            _cv2_preview("SynthText (img)", idict['img'])
                    viz_masks(2, idict['img'], seg, depth, regions['label'])
                except Exception:
                    _cv2_preview("SynthText (fallback)", idict['img'])

            res.append(idict.copy())

        return res
