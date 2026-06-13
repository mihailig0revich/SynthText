from typing import List, Optional, Tuple
import numpy as np
import cv2

def _ensure_masks(masks: Optional[List[np.ndarray]], shape: Tuple[int,int]) -> List[np.ndarray]:
    if masks is None:
        return []
    H, W = shape
    out = []
    for m in masks:
        m = (m > 0).astype(np.uint8) * 255
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append(m)
    return out

def _warp_affine(img, M, dsize, masks: List[np.ndarray]):
    img_out = cv2.warpAffine(img, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    masks_out = [
        cv2.warpAffine(m, M, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for m in masks
    ]
    return img_out, masks_out

def _warp_perspective(img, H, dsize, masks: List[np.ndarray]):
    img_out = cv2.warpPerspective(img, H, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    masks_out = [
        cv2.warpPerspective(m, H, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for m in masks
    ]
    return img_out, masks_out

def bboxes_from_masks(masks: List[np.ndarray]) -> List[Tuple[int,int,int,int]]:
    boxes = []
    for m in masks:
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            boxes.append((0, 0, 0, 0))
        else:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            boxes.append((x0, y0, x1, y1))
    return boxes

def aug_rotate(
    img: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    angle_deg: float = 5.0,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0
):
    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))
    if center is None:
        center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return _warp_affine(img, M, (W, H), masks)

def aug_perspective(
    img: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    max_jitter_ratio: float = 0.08
):
    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))

    src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    jx, jy = max_jitter_ratio*W, max_jitter_ratio*H
    dst = src + np.float32([
        [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)],
        [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)],
        [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)],
        [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)],
    ])

    Hmat = cv2.getPerspectiveTransform(src, dst)
    return _warp_perspective(img, Hmat, (W, H), masks)

def aug_gaussian_blur(img: np.ndarray, masks: Optional[List[np.ndarray]] = None, k: int = 3, sigma: float = 0.0):
    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    out = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)
    return out, masks

def aug_brightness_contrast_gamma(
    img: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 1.0
):
    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))
    x = img.astype(np.float32)

    x = cv2.convertScaleAbs(x, alpha=float(alpha), beta=float(beta))

    g = max(1e-6, float(gamma))
    x = np.clip((x.astype(np.float32) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)
    return x, masks

def aug_illumination_gradient(
    img: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    strength: float = 0.35,
    ellipse_prob: float = 0.6
):
    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))
    y, x = np.ogrid[:H, :W]
    cx, cy = np.random.uniform(0.2*W, 0.8*W), np.random.uniform(0.2*H, 0.8*H)

    if np.random.rand() < ellipse_prob:
        rx, ry = np.random.uniform(0.5*W, 1.2*W), np.random.uniform(0.5*H, 1.2*H)
        d = ((x - cx) ** 2) / (rx ** 2) + ((y - cy) ** 2) / (ry ** 2)
        mask = 1.0 - strength * d
    else:
        vx, vy = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        n = np.sqrt(vx * vx + vy * vy) + 1e-6
        vx, vy = vx / n, vy / n
        mask = 1.0 + strength * ((x - cx) * vx + (y - cy) * vy) / max(W, H)

    mask = np.clip(mask, 0.5, 1.5).astype(np.float32)
    out = np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)
    return out, masks

def aug_elastic(
    img: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    alpha: float = 30.0,
    sigma: float = 6.0
):
    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))

    rng = np.random.default_rng()
    dx = rng.standard_normal((H, W)).astype(np.float32)
    dy = rng.standard_normal((H, W)).astype(np.float32)

    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x = np.clip(xx + dx, 0, W - 1)
    map_y = np.clip(yy + dy, 0, H - 1)

    img_out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    masks_out = [
        cv2.remap(m, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for m in masks
    ]
    return img_out, masks_out

def aug_cutout(
    img: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    num_rects: int = 1,
    size_ratio: Tuple[float, float] = (0.1, 0.3),
    mode: str = "gray"
):
    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))
    out = img.copy()

    mean_col = out.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    for _ in range(num_rects):
        side = np.random.uniform(*size_ratio) * min(H, W)
        w = int(np.clip(np.random.uniform(0.6, 1.4) * side, 5, W))
        h = int(np.clip(side, 5, H))
        x0 = np.random.randint(0, max(1, W - w))
        y0 = np.random.randint(0, max(1, H - h))

        if mode == "gray":
            patch = np.full((h, w, 3), 127, np.uint8)
        elif mode == "noise":
            patch = np.random.randint(0, 256, (h, w, 3), np.uint8)
        elif mode == "mean":
            patch = np.tile(mean_col[None, None, :], (h, w, 1))
        elif mode == "copy":
            sx = np.random.randint(0, max(1, W - w))
            sy = np.random.randint(0, max(1, H - h))
            patch = out[sy:sy + h, sx:sx + w].copy()
        else:
            patch = np.full((h, w, 3), 127, np.uint8)

        out[y0:y0 + h, x0:x0 + w] = patch

    return out, masks

def apply_augmentations(
    img: np.ndarray,
    masks: Optional[List[np.ndarray]] = None,
    cfg: Optional[dict] = None
):
    if cfg is None:
        cfg = {}
    H, W = img.shape[:2]
    masks = _ensure_masks(masks, (H, W))
    out_img, out_masks = img, masks

    rnd = np.random.rand

    if rnd() < cfg.get("p_rotate", 0.5):
        ang = float(np.random.uniform(-8, 8))
        out_img, out_masks = aug_rotate(out_img, out_masks, angle_deg=ang)

    if rnd() < cfg.get("p_perspective", 0.35):
        out_img, out_masks = aug_perspective(out_img, out_masks, max_jitter_ratio=cfg.get("persp_jit", 0.08))

    if rnd() < cfg.get("p_elastic", 0.25):
        out_img, out_masks = aug_elastic(out_img, out_masks,
                                         alpha=cfg.get("elastic_alpha", 30.0),
                                         sigma=cfg.get("elastic_sigma", 6.0))

    if rnd() < cfg.get("p_blur", 0.35):
        k = int(np.random.choice([3, 5, 7]))
        out_img, out_masks = aug_gaussian_blur(out_img, out_masks, k=k, sigma=0.0)

    if rnd() < cfg.get("p_bc_gamma", 0.6):
        out_img, out_masks = aug_brightness_contrast_gamma(
            out_img, out_masks,
            alpha=np.random.uniform(0.75, 1.25),
            beta=np.random.uniform(-25, 25),
            gamma=np.random.uniform(0.8, 1.25),
        )
    if rnd() < cfg.get("p_illum", 0.4):
        out_img, out_masks = aug_illumination_gradient(out_img, out_masks, strength=np.random.uniform(0.2, 0.45))

    if rnd() < cfg.get("p_cutout", 0.35):
        out_img, out_masks = aug_cutout(
            out_img, out_masks,
            num_rects=int(np.random.randint(1, 3)),
            size_ratio=(0.12, 0.28),
            mode=np.random.choice(["gray", "noise", "mean", "copy"], p=[0.25, 0.25, 0.25, 0.25])
        )

    boxes = bboxes_from_masks(out_masks)
    return out_img, out_masks, boxes
