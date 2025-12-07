import numpy as np
import cv2
from typing import List, Optional, Tuple

# Если у тебя есть свои аугментации из transforms, импорт оставляем.
try:
    from transforms import aug_rotate, aug_perspective
except ImportError:
    aug_rotate = None
    aug_perspective = None


# ==========================
# БАЗОВЫЕ ХЕЛПЕРЫ
# ==========================
def _to_f01(img: np.ndarray) -> np.ndarray:
    """Перевод в float32 [0,1]. Терпит и uint8, и float."""
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    return x


def _to_u8(x: np.ndarray) -> np.ndarray:
    """Перевод из [0,1] в uint8, с отсечкой."""
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _ensure_masks(masks: Optional[List[np.ndarray]], shape: Tuple[int, int]) -> List[np.ndarray]:
    """Нормализуем список масок под размер картинки."""
    if masks is None:
        return []
    H, W = shape
    out: List[np.ndarray] = []
    for m in masks:
        if m is None:
            continue
        m = (m > 0).astype(np.uint8) * 255
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append(m)
    return out


# ==========================
# ПРОСТЫЕ ШУМЫ / ЭФФЕКТЫ
# ==========================
def noise_gaussian(img: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Усиленный гауссов шум, но с аккуратной отсечкой."""
    x = _to_f01(img)
    n = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return _to_u8(x + n)


def noise_speckle(img: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Усиленный speckle-шум (умножается на сигнал)."""
    x = _to_f01(img)
    n = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return _to_u8(x + x * n)

def adjust_contrast_time_of_day(img: np.ndarray, time_weight: float = None) -> np.ndarray:
    """
    Аккуратно меняет контраст в зависимости от времени суток.
    time_weight: 0.0 = ночь (низкий контраст), 0.5 = рассвет/закат, 1.0 = день (высокий контраст)
    Если None, выбирается случайное время суток.
    """
    if time_weight is None:
        # Случайное время суток: чаще вечер/ночь для реалистичности
        r = np.random.rand()
        if r < 0.4:    # ночь
            time_weight = 0.0
        elif r < 0.7:  # вечер/утро
            time_weight = 0.3 + np.random.rand() * 0.4
        else:          # день
            time_weight = 0.8 + np.random.rand() * 0.2

    x = _to_f01(img)
    H, W = x.shape[:2]

    # Базовый контраст для данного времени суток
    base_contrast = 0.7 + time_weight * 0.6  # ночь: ~0.7, день: ~1.3
    
    # Добавляем естественную вариативность
    contrast_factor = base_contrast * (0.95 + np.random.rand() * 0.1)
    
    # Применяем контраст с сохранением средней яркости
    mean = np.mean(x)
    x = (x - mean) * contrast_factor + mean
    x = np.clip(x, 0.0, 1.0)
    
    # Для ночного времени добавляем мягкий градиент контраста (центр ярче)
    if time_weight < 0.3:  # ночь
        cy, cx = H / 2.0, W / 2.0
        y, x_grid = np.ogrid[:H, :W]
        dist = np.sqrt(((x_grid - cx) / max(cx, 1.0))**2 + ((y - cy) / max(cy, 1.0))**2)
        # Центр имеет контраст ближе к дневному, края - более низкий
        local_contrast = 0.9 + (base_contrast - 0.9) * (1 - np.clip(dist, 0.0, 1.0)**1.5)
        local_contrast = local_contrast[..., np.newaxis]  # для RGB каналов
        
        x_center = (x - mean) * local_contrast + mean
        x = np.clip(x_center, 0.0, 1.0)
    
    return _to_u8(x)

def noise_saltpepper(img: np.ndarray, amount: float = 0.008, s_vs_p: float = 0.5) -> np.ndarray:
    """
    Соль-перец заметно сильнее: по умолчанию ~0.8% пикселей.
    Всё ещё далеко от «снега».
    """
    out = img.copy()
    h, w = out.shape[:2]
    num = int(amount * h * w)
    if num <= 0:
        return out

    # salt (белые точки)
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    out[ys, xs] = 255

    # pepper (чёрные точки)
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    out[ys, xs] = 0
    return out


def motion_blur(img: np.ndarray, k: int = 3, angle_deg: float = 0.0) -> np.ndarray:
    """Очень мягкий motion blur: маленькое ядро, слабый след."""
    k = int(max(1, min(k, 9)))
    if k == 1:
        return img
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s <= 1e-6:
        return img
    kernel /= s
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def vignette(img: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """
    Аккуратная виньетка: максимум ~10–15% затемнения по краям.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    # нормированное расстояние до центра
    dist2 = ((x - cx) / max(cx, 1.0)) ** 2 + ((y - cy) / max(cy, 1.0)) ** 2
    # mask в [1-strength, 1]
    mask = 1.0 - strength * np.clip(dist2, 0.0, 1.0)
    mask = mask.astype(np.float32)
    x_f = _to_f01(img)
    out = x_f * mask[..., None]
    return _to_u8(out)


def jpeg_compress_rgb(img: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    Прогон через JPEG, чтобы добавить артефакты.
    img ожидается в RGB.
    """
    if img is None or img.size == 0:
        return img

    quality = int(np.clip(quality, 5, 95))

    # OpenCV пишет JPEG в BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", bgr, encode_param)
    if not ok:
        return img

    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return out



# ==========================
# МЯГКИЙ COLOR JITTER
# ==========================
def color_jitter_rgb(
    img: np.ndarray,
    hue: float = 0.0,
    sat: float = 0.0,
    val: float = 0.0,
) -> np.ndarray:
    """
    Лёгкий цветовой джиттер:
    - hue: максимум сдвига тона (в градусах, 0..20 примерно)
    - sat: максимум относительного изменения насыщенности (0..0.3)
    - val: максимум относительного изменения яркости (0..0.3)
    """
    if img is None or img.size == 0:
        return img

    x = img.copy()
    # считаем, что у нас RGB (ты переводишь через to_rgb)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # сдвиг тона
    if hue > 1e-3:
        dh = np.random.uniform(-hue, hue)
        h = (h + dh) % 180.0

    # изменение насыщенности
    if sat > 1e-3:
        ds = np.random.uniform(-sat, sat)
        s = np.clip(s * (1.0 + ds), 0.0, 255.0)

    # изменение яркости
    if val > 1e-3:
        dv = np.random.uniform(-val, val)
        v = np.clip(v * (1.0 + dv), 0.0, 255.0)

    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out



# ==========================
# БАЗОВАЯ ДЕГРАДАЦИЯ СЦЕНЫ
# ==========================
def degrade_scene_rgb(
    img: np.ndarray,
    *,
    p_gauss=0.8,    # шум почти всегда
    p_speckle=0.8,  # зерно почти всегда
    p_sap=0.55,     # соль-перец на части кадров
    p_motion=0.3,   # blur чаще и сильнее
    p_vignette=0.45,
    p_color=0.75,
    p_jpeg=0.85,
    p_contrast=0.85,  # контраст почти всегда для времени суток
) -> np.ndarray:
    """
    Добавляет шумы и артефакты «плохой камеры» + имитацию времени суток.
    """
    out = img.copy()
    rnd = np.random.rand

    # Сначала контраст для времени суток
    if rnd() < p_contrast:
        out = adjust_contrast_time_of_day(out)

    # Гауссов шум: заметный, но не выжирает всё
    if rnd() < p_gauss:
        out = noise_gaussian(
            out,
            sigma=np.random.uniform(0.03, 0.06)  # немного усилено
        )

    # Speckle (мультипликативный) шум: чуть крупнее зерно
    if rnd() < p_speckle:
        out = noise_speckle(
            out,
            sigma=np.random.uniform(0.045, 0.085)  # немного усилено
        )

    # Соль-перец
    if rnd() < p_sap:
        out = noise_saltpepper(
            out,
            amount=np.random.uniform(0.04, 0.10),  # немного усилено
            s_vs_p=0.5,
        )

    # Motion blur – теперь чуть сильнее и чаще
    if rnd() < p_motion:
        out = motion_blur(
            out,
            k=np.random.randint(3, 8),  # увеличено до 8
            angle_deg=np.random.uniform(-25, 25)  # увеличен диапазон угла
        )

    # Дополнительная мягкая виньетка
    if rnd() < p_vignette:
        out = vignette(
            out,
            strength=np.random.uniform(0.10, 0.28)  # немного усилен диапазон
        )

    # Цветовой джиттер
    if rnd() < p_color:
        out = color_jitter_rgb(
            out,
            hue=np.random.uniform(5, 12),  # немного усилено
            sat=np.random.uniform(0.07, 0.15),  # немного усилено
            val=np.random.uniform(0.07, 0.15),  # немного усилено
        )

    # JPEG-артефакты
    if rnd() < p_jpeg:
        out = jpeg_compress_rgb(
            out,
            quality=np.random.randint(45, 88)  # немного снижен минимум
        )

    return out

# ==========================
# ГЛОБАЛЬНОЕ "ЗАТЕМНЕНИЕ"
# ==========================
def darken_scene_realistic(
    img: np.ndarray,
    strength_range=(0.8, 1.0),
    gamma_range=(0.95, 1.15),
    vignette_p: float = 0.5,
) -> np.ndarray:
    """
    Лёгкое реалистичное затемнение:
    - большинство кадров остаются близко к исходной яркости,
    - часть кадров чуть темнее,
    - виньетка иногда заметна, но не убивает картинку.
    """
    x = _to_f01(img)

    # 1) глобальное затемнение по фактору (0.8–1.0 от исходной яркости)
    mul = np.random.uniform(*strength_range)
    x = np.clip(x * mul, 0.0, 1.0)

    # 2) лёгкая гамма: чуть усиливаем тени, но без экстремального провала
    if gamma_range is not None:
        gamma = np.random.uniform(*gamma_range)
        x = np.power(x, gamma).astype(np.float32)
        x = np.clip(x, 0.0, 1.0)

    out = _to_u8(x)

    # 3) виньетка — реже и мягче
    if np.random.rand() < vignette_p:
        out = vignette(
            out,
            strength=np.random.uniform(0.12, 0.28)  # мягкие затемнённые края
        )

    return out




# ==========================
# "ПЛОХАЯ КАМЕРА" (СУПЕР-МЯГКО)
# ==========================
def noise_bad_camera_random(img: np.ndarray) -> np.ndarray:
    """
    Шумная «плохая камера».
    Шумов ощутимо больше, но без экстремального blur и тотального затемнения.
    """
    return degrade_scene_rgb(
        img,
        p_gauss=0.9,
        p_speckle=0.85,
        p_sap=0.6,
        p_motion=0.2,
        p_vignette=0.5,
        p_color=0.85,
        p_jpeg=0.9,
    )






# ==========================
# ГЛАВНАЯ ФУНКЦИЯ ДЛЯ SYNTHTEXT
# ==========================
def apply_random_augmentations(
    img: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    cfg: Optional[dict] = None
):
    """
    Локальные аугментации поверх SynthText-сцены.
    Усилены шумы, motion blur и добавлено время суток через контраст.
    """
    if cfg is None:
        cfg = {}

    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))
    out_img, out_masks = img, masks

    rnd = np.random.rand

    # Перспектива
    if rnd() < cfg.get("p_perspective", 0.35):
        out_img, out_masks = aug_perspective(
            out_img,
            out_masks,
            max_jitter_ratio=cfg.get("persp_jit", 0.06),
        )

    # Контраст времени суток
    if rnd() < cfg.get("p_contrast", 0.4):
        out_img = adjust_contrast_time_of_day(out_img)

    # Гауссов шум
    if rnd() < cfg.get("p_gauss", 0.75):  # немного увеличена вероятность
        out_img = noise_gaussian(
            out_img,
            sigma=np.random.uniform(0.02, 0.04)  # усилен диапазон
        )

    # Speckle шум
    if rnd() < cfg.get("p_speckle", 0.75):
        out_img = noise_speckle(
            out_img,
            sigma=np.random.uniform(0.035, 0.055)  # усилен диапазон
        )

    # Motion blur – оставляем сильно ограниченным, но немного усилен
    if rnd() < cfg.get("p_motion", 0.3):  # увеличена вероятность
        out_img = motion_blur(
            out_img,
            k=np.random.randint(3, 8),  # увеличен максимум
            angle_deg=np.random.uniform(-15, 15)  # увеличен диапазон
        )

    # Соль-перец
    if rnd() < cfg.get("p_sap", 0.65):  # немного увеличена вероятность
        out_img = noise_saltpepper(
            out_img,
            amount=np.random.uniform(0.04, 0.10)  # усилен диапазон
        )

    return out_img

