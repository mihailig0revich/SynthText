import cv2
import numpy as np


def warp_points(Hinv, pts_xy):
    """Map points from fronto-parallel coordinates to image coordinates."""
    pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    pts = np.concatenate([pts_xy, np.ones((len(pts_xy), 1), np.float32)], axis=1).T
    warped = np.asarray(Hinv, dtype=np.float32) @ pts
    warped /= warped[2:3, :] + 1e-6
    return warped[:2, :].T


def estimate_local_scale_grid(Hinv, free_mask_fp, k=9, delta=6, seed=None):
    """
    Estimate local FP->image scale for a homography.

    Returns a scalar scale in approximately image pixels per FP pixel.
    """
    if Hinv is None or free_mask_fp is None:
        return 1.0

    Hinv = np.asarray(Hinv, dtype=np.float64)
    if Hinv.shape != (3, 3) or (not np.isfinite(Hinv).all()):
        return 1.0

    mask = np.asarray(free_mask_fp).astype(np.uint8)
    if mask.ndim != 2:
        return 1.0

    height, width = mask.shape[:2]
    delta = int(max(1, delta))

    ys, xs = np.where(mask > 0)
    if xs.size < 16:
        return 1.0

    ok = (xs >= delta) & (xs < (width - delta)) & (ys >= delta) & (ys < (height - delta))
    xs = xs[ok]
    ys = ys[ok]
    if xs.size < 8:
        return 1.0

    if seed is not None:
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(xs.size, size=min(int(k), xs.size), replace=False)
    else:
        pick = np.random.choice(xs.size, size=min(int(k), xs.size), replace=False)

    x0 = xs[pick].astype(np.float32)
    y0 = ys[pick].astype(np.float32)

    p = np.stack([x0, y0], axis=1)
    px = np.stack([x0 + delta, y0], axis=1)
    py = np.stack([x0, y0 + delta], axis=1)

    def warp(pts):
        pts_cv = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts_cv, Hinv).reshape(-1, 2)

    try:
        w0 = warp(p)
        wx = warp(px)
        wy = warp(py)
    except Exception:
        return 1.0

    dx = np.linalg.norm(wx - w0, axis=1) / float(delta)
    dy = np.linalg.norm(wy - w0, axis=1) / float(delta)

    scale = 0.5 * (dx + dy)
    scale = scale[np.isfinite(scale)]
    if scale.size == 0:
        return 1.0

    out = float(np.median(scale))
    if not np.isfinite(out) or out <= 1e-6:
        return 1.0
    return out


def rescale_frontoparallel(p_fp, box_fp, p_im):
    """
    Rescale a fronto-parallel region to approximately match image-space size.
    """
    l1 = np.linalg.norm(box_fp[1, :] - box_fp[0, :])
    l2 = np.linalg.norm(box_fp[1, :] - box_fp[2, :])
    if l1 <= 1e-8 or l2 <= 1e-8:
        return 1.0

    n0 = np.argmin(np.linalg.norm(p_fp - box_fp[0, :][None, :], axis=1))
    n1 = np.argmin(np.linalg.norm(p_fp - box_fp[1, :][None, :], axis=1))
    n2 = np.argmin(np.linalg.norm(p_fp - box_fp[2, :][None, :], axis=1))

    if n0 < 0 or n1 < 0 or n2 < 0 or n0 >= len(p_im) or n1 >= len(p_im) or n2 >= len(p_im):
        return 1.0

    lt1 = np.linalg.norm(p_im[n1, :] - p_im[n0, :])
    lt2 = np.linalg.norm(p_im[n1, :] - p_im[n2, :])

    if np.isinf(lt1) or np.isinf(lt2) or np.isnan(lt1) or np.isnan(lt2):
        return 1.0
    if lt1 <= 1e-8 or lt2 <= 1e-8:
        return 1.0

    scale = max(lt1 / l1, lt2 / l2)
    if not np.isfinite(scale):
        scale = 1.0

    return scale


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return v * 0.0
    return v / norm


def rot3d_scaled(n_src, n_dst, strength=1.0, max_tilt_deg=None):
    """
    Rotate n_src toward n_dst with scaled strength instead of full alignment.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    n0 = normalize(n_src)
    n1 = normalize(n_dst)

    c = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
    angle = float(np.arccos(c))

    if max_tilt_deg is not None:
        max_tilt = float(np.deg2rad(max_tilt_deg))
        angle = min(angle, max_tilt)

    angle *= strength

    axis = np.cross(n0, n1)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8 or angle < 1e-8:
        return np.eye(3, dtype=np.float32)

    axis = axis / axis_norm
    x, y, z = axis.astype(np.float32)

    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )

    identity = np.eye(3, dtype=np.float32)
    sin_a = float(np.sin(angle))
    cos_a = float(np.cos(angle))

    rotation = identity + sin_a * skew + (1.0 - cos_a) * (skew @ skew)
    return rotation.astype(np.float32)
