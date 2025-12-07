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
    rgb:   HxWx3, uint8 (RGB или BGR - без разницы, геометрия только)
    depth: HxW, float32
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)

    H, W = rgb.shape[:2]
    cx = W * 0.5
    cy = H * 0.5

    # фокус по FOV
    f = 0.5 * W / np.tan(np.deg2rad(fov_deg) * 0.5)
    fx = fy = float(f)

    # Z из глубины
    Z = depth_to_z(depth, near=depth_near, far=depth_far, invert=invert_depth)

    # сетка пикселей
    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )

    # из пикселей в 3D
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).T  # (3, N)

    # поворот сцены
    R = euler_to_R(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)
    pts2 = R @ pts
    X2, Y2, Z2 = pts2[0, :], pts2[1, :], pts2[2, :]

    eps = 1e-4
    Z2 = np.maximum(Z2, eps)

    # обратно в пиксели
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
    Возвращает цвет текста (RGB 0–1), контрастный к bg_rgb.
    """
    import numpy as np
    for _ in range(50):
        txt_rgb = np.random.rand(3)
        if color_contrast(bg_rgb, txt_rgb) >= min_contrast:
            return txt_rgb
    # fallback — инвертируем

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
    minAspect = 0.3 # w > 0.3*h
    maxAspect = 7
    minArea = 100 # number of pix
    minWidth  = 24      # было 30
    minHeight = 24      # было 30
    pArea     = 0.55

    # RANSAC planar fitting params:
    dist_thresh = 0.10 # m
    num_inlier = 90
    ransac_fit_trials = 100
    min_z_projection = 0.25

    minW = 16

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
    def get_hw(pt,return_rot=False):
        pt = pt.copy()
        R = su.unrotate2d(pt)
        mu = np.median(pt,axis=0)
        pt = (pt-mu[None,:]).dot(R.T) + mu[None,:]
        h,w = np.max(pt,axis=0) - np.min(pt,axis=0)
        if return_rot:
            return h,w,R
        return h,w
 
    @staticmethod
    def filter(seg,area,label):
        """
        Apply the filter.
        The final list is ranked by area.
        """
        good = label[area > TextRegions.minArea]
        area = area[area > TextRegions.minArea]
        filt,R = [],[]
        for idx,i in enumerate(good):
            mask = seg==i
            xs,ys = np.where(mask)

            coords = np.c_[xs,ys].astype('float32')
            rect = cv2.minAreaRect(coords)          
            #box = np.array(cv2.cv.BoxPoints(rect))
            box = np.array(cv2.boxPoints(rect))
            h,w,rot = TextRegions.get_hw(box,return_rot=True)

            rect_area = max(1.0, float(w) * float(h))
            f = (
                h > TextRegions.minHeight * 0.8 and          # было жестче
                w > TextRegions.minWidth  * 0.8 and
                TextRegions.minAspect < (w / max(h, 1e-6)) < TextRegions.maxAspect and
                (float(area[idx]) / rect_area) >= (TextRegions.pArea * 0.85)  # было 0.60 → станет 0.51
            )
            filt.append(f)
            R.append(rot)

        # filter bad regions:
        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        aidx = np.argsort(-area)
        good = good[filt][aidx]
        R = [R[i] for i in aidx]
        filter_info = {'label':good, 'rot':R, 'area': area[aidx]}
        return filter_info

    @staticmethod
    def sample_grid_neighbours(mask,nsample,step=3):
        """
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.
        """
        if 2*step >= min(mask.shape[:2]):
            return #None

        y_m,x_m = np.where(mask)
        mask_idx = np.zeros_like(mask,'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i],x_m[i]] = i

        xp,xn = np.zeros_like(mask), np.zeros_like(mask)
        yp,yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:,:-2*step] = mask[:,2*step:]
        xn[:,2*step:] = mask[:,:-2*step]
        yp[:-2*step,:] = mask[2*step:,:]
        yn[2*step:,:] = mask[:-2*step,:]
        valid = mask&xp&xn&yp&yn

        ys,xs = np.where(valid)
        N = len(ys)
        if N==0: #no valid pixels in mask:
            return #None
        nsample = min(nsample,N)
        idx = np.random.choice(N,nsample,replace=False)
        # generate neighborhood matrix:
        # (1+4)x2xNsample (2 for y,x)
        xs,ys = xs[idx],ys[idx]
        s = step
        X = np.transpose(np.c_[xs,xs+s,xs+s,xs-s,xs-s][:,:,None],(1,2,0))
        Y = np.transpose(np.c_[ys,ys+s,ys-s,ys+s,ys-s][:,:,None],(1,2,0))
        sample_idx = np.concatenate([Y,X],axis=1)
        mask_nn_idx = np.zeros((5,sample_idx.shape[-1]),'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:,i] = mask_idx[sample_idx[:,:,i][:,0],sample_idx[:,:,i][:,1]]
        return mask_nn_idx

    @staticmethod
    def filter_depth(xyz,seg,regions):
        plane_info = {'label':[],
                      'coeff':[],
                      'support':[],
                      'rot':[],
                      'area':[]}
        for idx,l in enumerate(regions['label']):
            mask = seg==l
            pt_sample = TextRegions.sample_grid_neighbours(mask,TextRegions.ransac_fit_trials,step=3)
            if pt_sample is None:
                continue #not enough points for RANSAC
            # get-depths
            pt = xyz[mask]
            plane_model = su.isplanar(pt, pt_sample,
                                     TextRegions.dist_thresh,
                                     TextRegions.num_inlier,
                                     TextRegions.min_z_projection)
            if plane_model is not None:
                plane_coeff = plane_model[0]
                if np.abs(plane_coeff[2])>TextRegions.min_z_projection:
                    plane_info['label'].append(l)
                    plane_info['coeff'].append(plane_model[0])
                    plane_info['support'].append(plane_model[1])
                    plane_info['rot'].append(regions['rot'][idx])
                    plane_info['area'].append(regions['area'][idx])

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
    
    def estimate_perspective_from_depth_at_bbox(self, depth, bbox, max_alpha=0.9):
        """
        Более агрессивная оценка силы перспективы по локальному градиенту глубины.

        depth: (H, W) карта глубины (float32)
        bbox: (y0, x0, h, w) — окно, куда будет вставляться текст в IMG-координатах.
        max_alpha: максимальная "сырая" сила перспективы (0..~1).

        Возвращает alpha в [0, max_alpha]. Чем больше градиент глубины внутри bbox,
        тем сильнее перспектива.
        """
        import numpy as np, cv2

        if depth is None:
            return 0.0

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            return 0.0

        H, W = depth.shape[:2]
        y0, x0, h, w = bbox
        y1 = min(H, y0 + h)
        x1 = min(W, x0 + w)
        if y0 >= y1 or x0 >= x1:
            return 0.0

        patch = depth[y0:y1, x0:x1]
        if patch.size < 16:
            return 0.0

        dzdx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(dzdx * dzdx + dzdy * dzdy)

        valid = np.isfinite(grad)
        if not np.any(valid):
            return 0.0

        # Берём чуть более "жирный" хвост, чтобы не бояться шумов
        g_loc = float(np.percentile(grad[valid], 80))
        g_max = float(np.percentile(grad[valid], 95))
        if g_max <= 1e-6:
            return 0.0

        # Делаем ratio агрессивнее и чуть поджимаем сверху
        ratio = g_loc / g_max          # 0..~1
        ratio *= 1.8                   # усиливаем чувствительность
        ratio = min(ratio, 1.2)        # не даём улететь слишком далеко

        # Мягкая S-образная кривуля: слабые градиенты усиливаем, сильные поджимаем
        s = max(0.0, min(1.0, ratio))  # 0..1
        s = s ** 0.5                   # чем ближе к 1, тем сильнее

        return float(max_alpha * s)
    
    def warp_text_with_pseudo_perspective_at_bbox(self, text_img, bb_char_fp,
                                                  alpha, bbox_global, img_shape):
        """
        Делает более сильный псевдо-перспективный warp текста уже ПОСЛЕ выбора позиции.

        text_img: (h, w) маска текста в локальных координатах.
        bb_char_fp: (2, 4, N) bbox'ы символов в этих же локальных координатах.
        alpha: "сырая" сила эффекта (0..~1) из estimate_perspective_from_depth_at_bbox.
        bbox_global: (y0, x0, h, w) — куда этот патч встанет в изображение.
        img_shape: (H, W) исходного изображения.

        Усиления:
        - Чем дальше текст от центра кадра, тем сильнее перспектива.
        - Shift ограничен 70% ширины, чтобы прямоугольник не схлопнулся в линию.
        """
        import numpy as np, cv2

        if text_img is None or alpha is None or alpha <= 1e-3:
            return text_img, bb_char_fp

        h, w = text_img.shape[:2]
        if h < 2 or w < 2:
            return text_img, bb_char_fp

        H_img, W_img = img_shape
        y0, x0, h_box, w_box = bbox_global

        # Центр выбранного бокса и центр изображения
        cx = x0 + w_box * 0.5
        cy = y0 + h_box * 0.5
        cx_img = W_img * 0.5
        cy_img = H_img * 0.5

        dx = cx - cx_img
        dy = cy - cy_img

        # Нормируем расстояние от центра в [0..~1]
        dxn = dx / max(1.0, 0.5 * W_img)
        dyn = dy / max(1.0, 0.5 * H_img)
        d_norm = float(np.sqrt(dxn * dxn + dyn * dyn))
        d_norm = float(np.clip(d_norm, 0.0, 1.2))

        # Усиливаем alpha в зависимости от удаления от центра:
        # в центре ~0.7*alpha, на краях до ~1.5*alpha
        boost = 0.7 + 0.7 * d_norm
        alpha_eff = float(alpha * boost)
        alpha_eff = float(np.clip(alpha_eff, 0.0, 1.0))

        # Зона / "дальняя" сторона
        if abs(dx) >= abs(dy):
            zone = "horizontal"
            side_far = "left" if dx < 0 else "right"
        else:
            zone = "vertical"
            side_far = "top" if dy < 0 else "bottom"

        # Максимальный сдвиг — не больше 70% ширины текста
        shift = alpha_eff * float(w)
        shift = float(min(shift, 0.7 * float(w)))

        src = np.float32([
            [0.0,     0.0],      # TL
            [w - 1.0, 0.0],      # TR
            [w - 1.0, h - 1.0],  # BR
            [0.0,     h - 1.0],  # BL
        ])

        TL, TR, BR, BL = src.copy()

        if zone == "horizontal":
            if side_far == "left":
                TL = [shift, 0.0]
                BL = [shift, h - 1.0]
            else:
                TR = [w - 1.0 - shift, 0.0]
                BR = [w - 1.0 - shift, h - 1.0]
        else:
            if side_far == "top":
                TL = [0.0,      shift]
                TR = [w - 1.0,  shift]
            else:
                BL = [0.0,      h - 1.0 - shift]
                BR = [w - 1.0,  h - 1.0 - shift]

        dst = np.float32([TL, TR, BR, BL])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            text_img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if bb_char_fp is not None and bb_char_fp.size:
            pts = bb_char_fp.reshape(2, -1)
            ones = np.ones((1, pts.shape[1]), dtype=np.float32)
            pts_h = np.vstack([pts, ones])
            wpts = M @ pts_h
            wpts /= (wpts[2:3, :] + 1e-6)
            pts2 = wpts[:2, :]
            bb_char_fp_w = pts2.reshape(2, 4, bb_char_fp.shape[2])
        else:
            bb_char_fp_w = bb_char_fp

        return warped, bb_char_fp_w




    def filter_for_placement(self, xyz, seg, regions):
        """
        Строит place_mask / homography / homography_inv БЕЗ 3D-плоскостей
        и оставляет только самые большие сегменты.

        seg: карта сегментации (H, W), int
        regions: словарь после TextRegions.get_regions(...):
            'label', 'area', 'rot', ...
        """
        import numpy as np

        # аккуратно приводим seg к нормальному виду
        seg = np.nan_to_num(np.asarray(seg), nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
        H_img, W_img = seg.shape[:2]

        labels = np.asarray(regions.get("label", []), dtype=np.int32)
        areas  = np.asarray(regions.get("area", [])) if "area" in regions else None

        I = np.eye(3, dtype=np.float32)

        entries = []
        for idx, l in enumerate(labels):
            # маска региона по сегментации
            seg_mask = (seg == int(l)).astype(np.uint8)
            if seg_mask.sum() < 64:
                continue  # совсем мелкие выкидываем

            # place_mask: 0 = можно ставить текст, 255 = нельзя
            pm = np.where(seg_mask > 0, 0, 255).astype(np.uint8)

            # "полезная" площадь — сколько пикселей под текст (0)
            free_area = int(np.count_nonzero(pm == 0))
            if free_area <= 0:
                continue

            # если есть исходная area — используем её как основной вес,
            # иначе берём free_area
            if areas is not None and idx < len(areas):
                reg_area = float(areas[idx])
            else:
                reg_area = float(free_area)

            # сохраняем: (сортируемый_размер, свободная_площадь, label, маска)
            entries.append((reg_area, free_area, int(l), pm))

        # если ничего не набрали — фоллбек: почти весь кадр
        if not entries:
            margin = max(8, min(H_img, W_img) // 40)
            pm_full = 255 * np.ones((H_img, W_img), np.uint8)
            pm_full[margin:H_img - margin, margin:W_img - margin] = 0

            regions["place_mask"]     = [pm_full]
            regions["homography"]     = [I.copy()]
            regions["homography_inv"] = [I.copy()]
            regions["label"]          = np.array([1], dtype=np.int32)
            regions["area"]           = np.array([pm_full.size], dtype=np.float32)
            return regions

        # сортируем по площади региона по убыванию
        entries.sort(key=lambda x: x[0], reverse=True)

        # сколько оставить: ориентируемся на max_text_regions
        K = getattr(self, "max_text_regions", 4)
        # оставим не больше 2*K самых крупных регионов
        K = max(1, min(len(entries), K * 2))

        place_masks = []
        Hs, Hinvs   = [], []
        out_labels  = []
        out_areas   = []

        for reg_area, free_area, l, pm in entries[:K]:
            place_masks.append(pm)
            Hs.append(I.copy())
            Hinvs.append(I.copy())
            out_labels.append(l)
            out_areas.append(reg_area)

        regions["place_mask"]     = place_masks
        regions["homography"]     = Hs
        regions["homography_inv"] = Hinvs
        regions["label"]          = np.array(out_labels, dtype=np.int32)
        regions["area"]           = np.array(out_areas, dtype=np.float32)

        return regions

    
    def select_middle_regions_by_area(self, place_masks, K=6):
        """
        Вместо «около медианы» берём K регионов
        с максимальной площадью СВОБОДНОЙ части:
        (mask == 0) — туда можно ставить текст.
        Самый большой сегмент всегда будет первым.
        """
        import numpy as np

        if not place_masks:
            return []

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




    def select_region_for_text(self, txt_str, font, f_h_px_start, f_asp,
                               place_masks, regions, *,
                               gap_px=6, min_font_px=14, shrink_step=0.90,
                               side_margin=0.86, topK=4,
                               coarse_scale=0.4, coarse_step=4, fine_step=2,
                               min_char_px_img=8, fast_mode=False, img=None):
        """
        Выбор региона и размера шрифта + базового угла.
        5-й аргумент — список place_masks.
        """
        import numpy as np

        if not place_masks:
            return None, None, None

        if hasattr(self, "fast_mode"):
            fast_mode = bool(self.fast_mode)

        # кандидаты по площади (около медианы)
        cand = self.select_middle_regions_by_area(
            place_masks,
            K=(2 if fast_mode else topK),
        )

        if not cand:
            return None, None, None

        lo = max(int(min_font_px), 1)
        hi = int(max(f_h_px_start, lo))

        best_pair, candidate_angles, best_ireg, _ = self.find_best_region_for_text(
            txt_str, lo, hi, f_asp, place_masks, cand, regions,
            coarse_scale, coarse_step, fine_step, side_margin,
            fast_mode, shrink_step, min_char_px_img, img
        )
        if best_pair is None:
            return None, None, None

        ireg_best, f_best = best_pair

        # --- зона и финальный угол ---
        zone_type = "horizontal"
        dominant_orientation = 0.0
        img_shape = None
        text_height = 0
        text_width = 0
        region_coords_img = None

        if img is not None and ireg_best is not None:
            H_img, W_img = img.shape[:2]
            img_shape = (H_img, W_img)

            # центр региона в координатах изображения
            cy_img, cx_img = self.get_region_center_img(
                ireg_best, place_masks, regions, img_shape
            )

            # искусственный bbox вокруг центра — только для угловой логики
            reg_h = int(0.4 * H_img)
            reg_w = int(0.4 * W_img)
            y_min = max(0, min(H_img - 1, cy_img - reg_h // 2))
            x_min = max(0, min(W_img - 1, cx_img - reg_w // 2))
            if y_min + reg_h > H_img:
                reg_h = H_img - y_min
            if x_min + reg_w > W_img:
                reg_w = W_img - x_min
            region_coords_img = (y_min, x_min, reg_h, reg_w)

            zone_type = self.get_text_placement_zone(region_coords_img, img_shape)
            text_height, text_width = self.estimate_text_size(txt_str, f_best, f_asp)

            try:
                dominant_orientation = self.get_dominant_text_orientation(img, None)
            except Exception:
                dominant_orientation = 0.0

        if candidate_angles:
            selected_angle = self.select_optimal_angle(
                candidate_angles,
                dominant_orientation,
                zone_type,
                region_coords_img,
                img_shape,
                text_height,
                text_width,
            )
        else:
            selected_angle = 0.0

        return ireg_best, float(f_best), float(selected_angle)

    def render_text_overlay_perspective(self, img, txt_str, font,
                                        selected_angle=None,
                                        place_mask=None,
                                        H=None, Hinv=None,
                                        depth=None,
                                        occupied_global=None,
                                        gap_px=6):
        """
        Рисует текст в фронто-параллельной маске региона и
        проецирует его в изображение через гомографию Hinv.
        selected_angle задаёт ориентацию в плоскости региона.
        """
        import numpy as np, cv2

        if place_mask is None or H is None or Hinv is None:
            return None, None, None

        try:
            txt_arr, txt_str_norm, bb_char_xywh = self.text_renderer.render_curved(
                font, txt_str,
                char_gap_px=getattr(self, "char_gap_px", 0),
                word_gap_px=getattr(self, "word_gap_px", 1),
            )
        except Exception:
            return None, None, None

        if txt_arr is None or txt_arr.size == 0:
            return None, None, None

        bb_char_fp = self.text_renderer.bb_xywh2coords(bb_char_xywh)

        pad_t, pad_b, pad_l, pad_r = 1, max(2, int(0.18 * font.size)), 1, 1
        txt_arr = cv2.copyMakeBorder(
            txt_arr, pad_t, pad_b, pad_l, pad_r,
            borderType=cv2.BORDER_CONSTANT, value=0,
        )
        bb_char_fp[0, :, :] += pad_l
        bb_char_fp[1, :, :] += pad_t

        if selected_angle is not None and abs(selected_angle) > 1e-3:
            txt_arr, bb_char_fp = self.rotate_text_and_bbox(
                txt_arr, bb_char_fp, selected_angle
            )

        h_txt, w_txt = txt_arr.shape[:2]
        H_fp, W_fp = place_mask.shape[:2]

        # строим "запрещённую" маску с учётом занятости и глубины
        try:
            forbid_fp = build_forbidden_mask_rectified(
                place_mask, H, occupied_global, depth, gap=gap_px
            )
        except Exception:
            forbid_fp = place_mask.copy()

        free255 = (forbid_fp == 0).astype(np.uint8) * 255

        ok, top_left = any_fit_with_integral(free255, h_txt, w_txt, step=2)
        if (not ok) or (top_left is None):
            return None, None, None

        y0_fp, x0_fp = top_left
        text_mask_fp = np.zeros_like(place_mask, dtype=np.uint8)

        mask_local = (txt_arr > 0).astype(np.uint8) * 255
        h_clip = min(h_txt, max(0, H_fp - y0_fp))
        w_clip = min(w_txt, max(0, W_fp - x0_fp))
        if h_clip <= 0 or w_clip <= 0:
            return None, None, None

        text_mask_fp[y0_fp:y0_fp + h_clip, x0_fp:x0_fp + w_clip] = \
            mask_local[:h_clip, :w_clip]

        bb_char_fp[0, :, :] += x0_fp
        bb_char_fp[1, :, :] += y0_fp

        H_img, W_img = img.shape[:2]
        warped_mask = cv2.warpPerspective(
            text_mask_fp, Hinv, (W_img, H_img),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
        )

        N = bb_char_fp.shape[2]
        pts_fp  = bb_char_fp.reshape(2, 4 * N).T.astype(np.float32)  # (4N,2)
        pts_img = _warp_points(Hinv, pts_fp)
        bb_img  = pts_img.T.reshape(2, 4, N)

        min_h = self.get_min_h(bb_img, txt_str_norm)
        img_colored = self.colorizer.color(
            img.copy(), [warped_mask], np.array([min_h])
        )

        return img_colored, bb_img, warped_mask




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
    
    def warp_text_with_pseudo_perspective(self, text_img, bb_char_fp,
                                          alpha, region_coords, img_shape):
        """
        Делает лёгкий перспективный варп текста:
        - alpha в [0..~0.35] задаёт "силу" перспективы (чем больше, тем сильнее трапеция),
        - направление берётся из положения region_coords относительно центра изображения.

        Мы НЕ используем никакие H/Hinv, только warpPerspective по трапеции.
        """
        import numpy as np, cv2

        if text_img is None or alpha is None or alpha <= 1e-3:
            return text_img, bb_char_fp

        h, w = text_img.shape[:2]
        if h < 2 or w < 2:
            return text_img, bb_char_fp

        H_img, W_img = img_shape
        y_min, x_min, reg_h, reg_w = region_coords
        # центр региона
        cy = y_min + reg_h * 0.5
        cx = x_min + reg_w * 0.5
        # центр изображения
        cy_img = H_img * 0.5
        cx_img = W_img * 0.5

        # горизонтальная / вертикальная зона
        zone_type = self.get_text_placement_zone(region_coords, (H_img, W_img))

        # от какой стороны "сжимать"
        side_horiz = "right" if cx >= cx_img else "left"
        side_vert  = "bottom" if cy >= cy_img else "top"

        # Насколько сильно можно сжать сторону
        shift = float(alpha) * w  # max ~ 35% ширины

        # исходный прямоугольник текста
        src = np.float32([
            [0.0, 0.0],     # TL
            [w - 1.0, 0.0], # TR
            [w - 1.0, h-1.0],# BR
            [0.0, h-1.0],   # BL
        ])

        # целевые точки (трапеция)
        TL, TR, BR, BL = src.copy()

        if zone_type == "horizontal":
            if side_horiz == "left":
                # регион слева → левая сторона дальше → сжимаем левую грань
                TL = [shift, 0.0]
                BL = [shift, h - 1.0]
            else:
                # регион справа → правая сторона дальше → сжимаем правую грань
                TR = [w - 1.0 - shift, 0.0]
                BR = [w - 1.0 - shift, h - 1.0]
        else:
            # вертикальная зона
            if side_vert == "top":
                # сверху → верхняя сторона дальше
                TL = [0.0, shift]
                TR = [w - 1.0, shift]
            else:
                # снизу → нижняя сторона дальше
                BL = [0.0, h - 1.0 - shift]
                BR = [w - 1.0, h - 1.0 - shift]

        dst = np.float32([TL, TR, BR, BL])

        # матрица перспективного преобразования
        M = cv2.getPerspectiveTransform(src, dst)

        # варп текста
        warped = cv2.warpPerspective(
            text_img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # варп bbox (2 x 4 x N)
        if bb_char_fp is not None and bb_char_fp.size:
            pts = bb_char_fp.reshape(2, -1)          # (2, 4*N)
            ones = np.ones((1, pts.shape[1]), dtype=np.float32)
            pts_h = np.vstack([pts, ones])           # (3, 4*N)
            wpts = M @ pts_h                         # (3, 4*N)
            wpts /= (wpts[2:3, :] + 1e-6)
            pts2 = wpts[:2, :]
            bb_char_fp_warped = pts2.reshape(2, 4, bb_char_fp.shape[2])
        else:
            bb_char_fp_warped = bb_char_fp

        return warped, bb_char_fp_warped

    
    def estimate_perspective_strength_from_depth(self, depth, region_mask, max_alpha=0.35):
        """
        depth: 2D карта глубины (H x W, float)
        region_mask: маска региона (H x W), >0 внутри региона

        Возвращает alpha в [0, max_alpha], которая говорит,
        насколько сильно искажать текст под "перспективу".
        Чем больше локальный градиент глубины — тем больше alpha.
        """
        import numpy as np, cv2

        if depth is None or region_mask is None:
            return 0.0

        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape[:2] != region_mask.shape[:2]:
            return 0.0

        # Градиент по всей карте глубины
        dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(dzdx * dzdx + dzdy * dzdy)

        valid = np.isfinite(grad_mag)
        if not np.any(valid):
            return 0.0

        # Значения градиента внутри региона
        mask = (region_mask > 0) & valid
        vals = grad_mag[mask]
        if vals.size < 32:
            return 0.0

        # Локальная "типичная" величина наклона
        g_loc = np.percentile(vals, 75)

        # Глобальный "верхний" уровень наклона, чтобы нормировать
        g_all = np.percentile(grad_mag[valid], 95)
        if g_all <= 1e-6:
            return 0.0

        strength = float(np.clip(g_loc / g_all, 0.0, 1.0))
        return strength * float(max_alpha)

    
    def place_text_textfirst(self, img, place_masks, regions, gap=6,
                             min_font_px=14, start_font_px=None, start_font_px_range=None,
                             shrink_step=0.90, depth=None, occupied_global=None):
        """
        Версия с размещением текста СТРОГО внутри выбранного сегмента.
        Возвращает (img_new, text, charBB, warped_mask).
        """
        import numpy as np

        H_img, W_img = img.shape[:2]

        if not place_masks:
            return None

        if 'segmentation' not in regions:
            regions['segmentation'] = place_masks

        # --- шрифт ---
        fs = self.text_renderer.font_state.sample()
        font = self.text_renderer.font_state.init_font(fs)
        f_asp = self.text_renderer.font_state.get_aspect_ratio(font)

        short_side = float(min(H_img, W_img))
        base_char = max(min_font_px, short_side / 14.0)

        if start_font_px_range is not None:
            lo_rng, hi_rng = start_font_px_range
            if lo_rng > hi_rng:
                lo_rng, hi_rng = hi_rng, lo_rng
            f_start = int(np.random.randint(int(lo_rng), int(hi_rng) + 1))
            f_start = max(f_start, int(min_font_px))
        elif start_font_px is not None:
            f_start = max(float(start_font_px), float(min_font_px))
        else:
            last = getattr(self, "last_font_h_px", None)
            if last is not None:
                f_start = max(float(last), base_char)
            else:
                f_start = base_char

        jitter = np.random.uniform(0.9, 1.1)
        f_start *= jitter
        f_start = float(np.clip(f_start, min_font_px, short_side / 16.0))

        f_layout = int(round(f_start * 1.30))
        if f_layout < min_font_px:
            f_layout = int(min_font_px)

        # --- оценка (nline, nchar) от get_nline_nchar + форс минимального nchar ---
        try:
            nline_raw, nchar_raw = self.text_renderer.get_nline_nchar(
                (128, 512),
                f_layout,
                f_layout * f_asp
            )
        except Exception:
            nline_raw, nchar_raw = 1, 12

        # форсим разумные значения
        nline_eff = max(1, int(nline_raw or 1))

        if nchar_raw is None or int(nchar_raw) < 6:
            nchar_eff = 10
        else:
            nchar_eff = int(nchar_raw)

        # --- надёжный выбор текста ---
        txt_str = self._sample_layout_text(nline_eff, nchar_eff, max_retries=5)

        if not txt_str or not isinstance(txt_str, str) or not txt_str.strip():
            return None

        txt_str = txt_str.strip()

        # --- выбираем регион и итоговый размер ---
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
            img=img
        )
        if ireg is None:
            return None

        key = (int(ireg), int(round(f_fit)))
        if hasattr(self, "_failed_pairs") and (key in self._failed_pairs):
            return None

        # --- проверка локального масштаба ---
        try:
            Hinv_reg = regions['homography_inv'][ireg]
            free_mask_fp = (place_masks[ireg] == 0).astype(np.uint8)
            s_loc = estimate_local_scale_grid(Hinv_reg, free_mask_fp, k=9, delta=6)
            if s_loc * f_fit < getattr(self, "min_char_px_img", 8):
                return None
        except Exception:
            pass

        font.size = self.text_renderer.font_state.get_font_size(font, f_fit)

        region_mask = place_masks[ireg]
        region_coords = self.get_region_coordinates(region_mask)
        if region_coords is None:
            return None

        img_new, bb_img, text_mask_img = self.render_text_overlay(
            img, txt_str, font,
            selected_angle=selected_angle,
            region_coords=region_coords,
            depth=depth
        )

        if img_new is None:
            return None

        self.last_font_h_px = f_fit

        return img_new, txt_str, bb_img, text_mask_img



    def get_region_coordinates(self, region_mask):
        """
        Получает координаты bounding box для свободного пространства в регионе.
        Возвращает (y_min, x_min, height, width) или None если регион пустой.
        """
        # Находим координаты свободного пространства (где маска == 0)
        free_space_coords = np.column_stack(np.where(region_mask == 0))
        
        if len(free_space_coords) == 0:
            return None
        
        # Находим границы свободного пространства
        y_coords = free_space_coords[:, 0]
        x_coords = free_space_coords[:, 1]
        
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        
        return (y_min, x_min, height, width)
    
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
        Надёжный выбор текста под заданные (nline, nchar):
        - несколько попыток text_source.sample(...)
        - нормализация списка -> строки
        - проверка на непустоту
        - fallback-текст, если всё совсем плохо
        """
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

            if txt_str:
                return txt_str

        # --- Fallback — чтобы генерация вообще не умирала ---
        fallback = "пример текста для генерации"
        return fallback




    def render_text_overlay(self, img, txt_str, font,
                            selected_angle=None,
                            region_coords=None,
                            depth=None):
        """
        Рисует текст внутри region_coords:
          1) рендер текста в локальный патч,
          2) поворот текста,
          3) выбор (y0, x0) внутри region_coords с ЗАПРЕТОМ сектора 30–150°,
          4) по depth[y0:y1, x0:x1] считаем силу перспективы (если depth есть),
          5) псевдо-перспективный warp патча,
          6) вклейка в img,
          7) усиление яркости текста + тёмная обводка.
        """
        import numpy as np, cv2

        H, W = img.shape[:2]

        # 1) рендер текста в фронтопараллельном виде
        try:
            txt_arr, txt_str_norm, bb_char_xywh = self.text_renderer.render_curved(
                font, txt_str,
                char_gap_px=getattr(self, "char_gap_px", 0),
                word_gap_px=getattr(self, "word_gap_px", 1),
            )
        except Exception:
            return None, None, None

        if txt_arr is None or txt_arr.size == 0:
            return None, None, None

        bb_char_fp = self.text_renderer.bb_xywh2coords(bb_char_xywh)

        # 2) паддинги вокруг текста
        pad_t = 1
        pad_b = max(2, int(0.18 * font.size))
        pad_l, pad_r = 1, 1
        txt_arr = cv2.copyMakeBorder(
            txt_arr, pad_t, pad_b, pad_l, pad_r,
            borderType=cv2.BORDER_CONSTANT, value=0,
        )
        bb_char_fp[0, :, :] += pad_l
        bb_char_fp[1, :, :] += pad_t

        h_fp, w_fp = txt_arr.shape[:2]

        # 3) поворот всего патча с bbox
        if selected_angle is not None and abs(selected_angle) > 1e-3:
            txt_arr, bb_char_fp = self.rotate_text_and_bbox(
                txt_arr, bb_char_fp, selected_angle
            )
            h_fp, w_fp = txt_arr.shape[:2]

        # 4) если патч > изображения — чуть ужмём
        if h_fp >= H or w_fp >= W:
            scale = min(
                (H - 2) / max(h_fp, 1),
                (W - 2) / max(w_fp, 1),
                0.85
            )
            if not (np.isfinite(scale) and scale > 0):
                return None, None, None

            new_w = max(1, int(round(w_fp * scale)))
            new_h = max(1, int(round(h_fp * scale)))
            txt_arr = cv2.resize(txt_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            sx = new_w / max(w_fp, 1e-6)
            sy = new_h / max(h_fp, 1e-6)
            bb_char_fp[0, :, :] *= sx
            bb_char_fp[1, :, :] *= sy
            h_fp, w_fp = new_h, new_w

        # 5) рабочий регион в координатах IMG
        if region_coords is None:
            y_region, x_region, reg_h, reg_w = 0, 0, H, W
        else:
            y_region, x_region, reg_h, reg_w = region_coords

        max_y_start = y_region + reg_h - h_fp
        min_y_start = y_region
        max_x_start = x_region + reg_w - w_fp
        min_x_start = x_region

        if min_y_start > max_y_start or min_x_start > max_x_start:
            return None, None, None

        # 6) выбор (y0, x0) c ЗАПРЕТОМ сектора 30–150° от центра кадра
        cx_img = W * 0.5
        cy_img = H * 0.5

        max_tries = 10
        accepted = False
        y0 = x0 = None

        for _ in range(max_tries):
            y0_try = np.random.randint(min_y_start, max_y_start + 1)
            x0_try = np.random.randint(min_x_start, max_x_start + 1)

            cx_txt = x0_try + 0.5 * w_fp
            cy_txt = y0_try + 0.5 * h_fp

            dx = cx_txt - cx_img
            dy_math = cy_img - cy_txt  # "математическое" Y вверх

            angle = np.degrees(np.arctan2(dy_math, dx))  # (-180, 180]
            if angle < 0.0:
                angle += 360.0  # [0, 360)

            # 🔴 Запрещённый сектор: 30–150° (верх + чуть боков)
            if 30.0 <= angle <= 150.0:
                continue

            y0, x0 = int(y0_try), int(x0_try)
            accepted = True
            break

        if not accepted:
            return None, None, None

        # 7) оценка перспективы ИМЕННО в этом месте
        bbox_global = (int(y0), int(x0), int(h_fp), int(w_fp))
        alpha = 0.0
        if depth is not None:
            try:
                alpha = self.estimate_perspective_from_depth_at_bbox(
                    depth, bbox_global, max_alpha=0.9
                )
            except Exception:
                alpha = 0.0

        if alpha > 1e-3:
            txt_arr, bb_char_fp = self.warp_text_with_pseudo_perspective_at_bbox(
                txt_arr, bb_char_fp,
                alpha=alpha,
                bbox_global=bbox_global,
                img_shape=(H, W),
            )
            h_fp, w_fp = txt_arr.shape[:2]

        # 8) собираем маску текста в координатах IMG
        text_mask_img = np.zeros((H, W), np.uint8)

        end_y = min(y0 + h_fp, H)
        end_x = min(x0 + w_fp, W)
        h_to_copy = end_y - y0
        w_to_copy = end_x - x0

        if h_to_copy <= 0 or w_to_copy <= 0:
            return None, None, None

        mask_fp = (txt_arr > 0).astype(np.uint8) * 255
        text_mask_img[y0:end_y, x0:end_x] = np.maximum(
            text_mask_img[y0:end_y, x0:end_x],
            mask_fp[:h_to_copy, :w_to_copy],
        )

        nz_imgmask = int((text_mask_img > 0).sum())
        if nz_imgmask == 0:
            return None, None, None

        # 9) переносим bbox в IMG-координаты
        bb_img = bb_char_fp.copy()
        bb_img[0, :, :] += x0
        bb_img[1, :, :] += y0

        # 10) базовая раскраска через Colorize
        try:
            min_h = self.get_min_h(bb_img, txt_str_norm)
        except Exception:
            return None, None, None

        try:
            img_colored = self.colorizer.color(
                img.copy(), [text_mask_img], np.array([min_h])
            )
        except Exception:
            return None, None, None

        text_region = (text_mask_img > 0)
        nz_text_region = int(text_region.sum())

        # 11) Усиливаем яркость текста + тёмная обводка
        if nz_text_region > 0:
            img_f = img_colored.astype(np.float32)

            # яркость / "неон"
            bright_factor = 1.5
            img_f[text_region] *= bright_factor
            img_f = np.clip(img_f, 0, 255)
            img_colored = img_f.astype(np.uint8)

            # обводка
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dil = cv2.dilate(text_mask_img, kernel, 1)
            er = cv2.erode(text_mask_img, kernel, 1)
            outline = ((dil > 0) & (er == 0))

            if int(outline.sum()) > 0:
                img_f = img_colored.astype(np.float32)
                dark_factor = 0.25
                img_f[outline] *= dark_factor
                img_colored = np.clip(img_f, 0, 255).astype(np.uint8)

        # Финальный "неонный" буст
        img_colored = self._boost_text_neon(img_colored, text_mask_img)

        return img_colored, bb_img, text_mask_img



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

    def get_num_text_regions(self, nregions):
        #return nregions
        nmax = min(self.max_text_regions, nregions)
        if np.random.rand() < 0.10:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(5.0,1.0)
        return int(np.ceil(nmax * rnd))

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



    # synthgen.py — внутри class RendererV3
        # synthgen.py — внутри class RendererV3
    def render_text(self, rgb, depth, seg, area, label, ninstance=1, viz=False):
        """
        Основной генератор:
        1) строит регионы и их place_mask,
        2) несколько раз пытается разместить текст по стратегии text-first,
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
            # Регионы по сегментации (геометрию по глубине мы уже отключили)
            regions = TextRegions.get_regions(None, seg, area, label)

            # Превращаем их в place_mask + единичные H/Hinv
            regions = self.filter_for_placement(None, seg, regions)
            if len(regions.get('place_mask', [])) == 0:
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

            nregions = len(regions['place_mask'])
            if nregions < 1:
                return []

            # сколько блоков хотим в среднем
            target_blocks = self.get_num_text_regions(nregions)

            # динамические бюджеты (сколько попыток на картинку и на регион)
            global_budget, per_region_cap, max_shrink_trials = self._compute_budgets(
                nregions, target_blocks
            )
            self.global_attempt_budget = global_budget
            self.per_region_attempt_cap = per_region_cap
            self._max_shrink_trials_runtime = max_shrink_trials
        except Exception:
            return []

        res = []
        H0, W0 = rgb.shape[:2]

        for i in range(ninstance):
            img = rgb.copy()
            itext, ibb = [], []

            # глобальная карта занятости в координатах изображения
            occupied_global = np.zeros(img.shape[:2], dtype=np.uint8)

            # сколько текстовых блоков хотим в этом экземпляре
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
                itext.append(text)
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



