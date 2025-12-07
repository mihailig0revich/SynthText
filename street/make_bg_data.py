import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List

import numpy as np
import cv2
import h5py
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2lab
from tqdm import tqdm

try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __enter__(self): return self
        def __exit__(self, *a): return False


CONFIG = SimpleNamespace(
    # вход / выход
    images=r"C:\code\SynthText-python3\street\img",
    out=r"C:\code\SynthText-python3\street\bg_data",

    # MiDaS
    device="cuda",              # "cuda" | "cpu" | "auto"
    midas_model="MiDaS_small",  # "DPT_Large" | "DPT_Hybrid" | "MiDaS_small"

    # лимит картинок (0 = без лимита)
    max_images=0,

    # depth postprocess (минимальный)
    pmin=5.0,
    pmax=95.0,
    gamma=0.9,

    # простая Felzenszwalb-сегментация
    felz_scale=120.0,
    felz_sigma=0.8,
    felz_min_size=900,
    felz_depth_w=0.2,
    felz_down=0.5,   # во сколько раз уменьшать картинку перед felzenszwalb
)


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


def _h5_safe_name(name: str) -> str:
    import re
    s = name.strip()
    s = s.replace("/", "_")
    s = re.sub(r"[^0-9A-Za-zА-Яа-я_\-\.]+", "_", s)
    return s or "sample"


def compute_areas_from_seg(seg_1based: np.ndarray) -> np.ndarray:
    labels, counts = np.unique(seg_1based, return_counts=True)
    if labels.size == 0:
        return np.zeros(1, dtype=np.int32)
    K = int(labels.max())
    area = np.zeros(max(K, 1), dtype=np.int32)
    area[labels.astype(int) - 1] = counts.astype(np.int32)
    return area


def save_to_h5_split(h5_path: str, name: str,
                     bgr: np.ndarray,
                     depth01: np.ndarray,
                     seg_1based: np.ndarray) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    depth_16bit = np.clip(depth01, 0.0, 1.0) * 65535.0
    depth_16bit = depth_16bit.astype(np.uint16)

    seg_i16 = seg_1based.astype(np.uint16)
    area = compute_areas_from_seg(seg_i16)

    key = _h5_safe_name(name)

    with h5py.File(h5_path, "a") as f:
        g_img   = f.require_group("img")
        g_depth = f.require_group("depth")
        g_seg   = f.require_group("seg")
        g_area  = f.require_group("area")

        for g in (g_img, g_depth, g_seg, g_area):
            if key in g:
                del g[key]

        g_img.create_dataset(key,   data=rgb,        compression="gzip", compression_opts=4, dtype="uint8")
        g_depth.create_dataset(key, data=depth_16bit, compression="gzip", compression_opts=4, dtype="uint16")
        g_seg.create_dataset(key,   data=seg_i16,     compression="gzip", compression_opts=4, dtype="uint16")
        g_area.create_dataset(key,  data=area,        compression="gzip", compression_opts=4, dtype="int32")


def depth_postprocess(depth01: np.ndarray,
                      pmin: float = 5.0,
                      pmax: float = 95.0,
                      gamma: float = 0.9) -> np.ndarray:
    """Очень быстрый и простой постпроцесс depth-карты."""
    d = depth01.astype(np.float32)
    d = np.nan_to_num(d, nan=np.nanmedian(d))
    d = np.clip(d, 0.0, 1.0)

    lo = np.percentile(d, pmin)
    hi = np.percentile(d, pmax)
    if hi > lo:
        d = (d - lo) / (hi - lo)
    d = np.clip(d, 0.0, 1.0)

    d = np.power(d, max(1e-3, float(gamma)))
    return np.clip(d, 0.0, 1.0)


class MidasDepthPredictor:
    def __init__(self, device: str = "cuda", model_name: str = "MiDaS_small"):
        import torch
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[INFO] Загрузка MiDaS ({model_name})...", flush=True)
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_name in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        # небольшой прогрев
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        _ = self.predict(dummy)
        print("[INFO] MiDaS готов.", flush=True)

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        import torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            pred = self.model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy().astype("float32")

        d = pred.copy()
        d -= d.min()
        maxv = d.max()
        if maxv > 0:
            d /= maxv
        return d

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        return self.predict(bgr)


def _build_features_lab_depth(bgr: np.ndarray,
                              depth01: Optional[np.ndarray],
                              depth_w: float) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = rgb2lab(rgb).astype(np.float32)
    L = np.clip(lab[..., 0] / 100.0, 0, 1)
    A = (lab[..., 1] + 128.0) / 255.0
    B = (lab[..., 2] + 128.0) / 255.0
    if depth01 is None or depth_w <= 1e-6:
        return np.stack([L, A, B], axis=-1)
    D = np.clip(depth01.astype(np.float32), 0, 1)
    return np.stack([L, A, B, depth_w * D], axis=-1)


def _renumber_1based(seg_any: np.ndarray) -> np.ndarray:
    seg_any = seg_any.astype(np.int32, copy=False)
    uniq = np.unique(seg_any)
    remap = {int(lbl): i + 1 for i, lbl in enumerate(uniq)}
    out = np.zeros_like(seg_any, dtype=np.int32)
    for old, new in remap.items():
        out[seg_any == old] = new
    return out


def segment_fast_felz(bgr: np.ndarray,
                      depth01: Optional[np.ndarray],
                      scale: float,
                      sigma: float,
                      min_size: int,
                      depth_w: float,
                      down: float) -> np.ndarray:
    feat = _build_features_lab_depth(bgr, depth01, depth_w)

    H, W = feat.shape[:2]
    if 0.2 <= float(down) < 1.0:
        h2, w2 = int(H * down), int(W * down)
        feat_small = cv2.resize(feat, (w2, h2), interpolation=cv2.INTER_LINEAR)
        seg_small = felzenszwalb(
            feat_small, scale=scale, sigma=sigma,
            min_size=min_size, channel_axis=-1
        )
        seg_full = cv2.resize(
            seg_small.astype(np.int32), (W, H),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        seg_full = felzenszwalb(
            feat, scale=scale, sigma=sigma,
            min_size=min_size, channel_axis=-1
        ).astype(np.int32)

    return _renumber_1based(seg_full)


def main():
    cfg = CONFIG

    img_dir = Path(cfg.images)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([
        p for p in img_dir.glob("*.*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if not images:
        print(f"[WARN] В папке {img_dir} нет изображений.")
        return

    print("[INFO] Инициализация MiDaS...")
    depth_predictor = MidasDepthPredictor(
        device=cfg.device,
        model_name=cfg.midas_model,
    )

    h5_path = str(out_dir / "bg_data.h5")
    names: List[str] = []

    max_images = int(cfg.max_images or 0)

    for i, img_path in enumerate(tqdm(images, desc="BG data", unit="img")):
        if max_images and i >= max_images:
            break

        bgr = imread_unicode(str(img_path))
        if bgr is None:
            print(f"[WARN] не удалось прочитать {img_path}")
            continue

        # depth
        depth = depth_predictor(bgr)
        depth = 1.0 - depth  # как в исходном коде
        depth = depth_postprocess(
            depth,
            pmin=cfg.pmin,
            pmax=cfg.pmax,
            gamma=cfg.gamma,
        )

        # segmentation
        seg = segment_fast_felz(
            bgr=bgr,
            depth01=depth,
            scale=cfg.felz_scale,
            sigma=cfg.felz_sigma,
            min_size=int(cfg.felz_min_size),
            depth_w=float(cfg.felz_depth_w),
            down=float(cfg.felz_down),
        )

        if seg.size == 0:
            print(f"[WARN] пустая сегментация для {img_path.name}")
            continue

        base = img_path.stem
        save_to_h5_split(h5_path, base, bgr, depth, seg)
        names.append(base)

    # список имён
    if names:
        with h5py.File(h5_path, "a") as f:
            meta = f.require_group("meta")
            if "names" in meta:
                del meta["names"]
            maxlen = max(len(n) for n in names)
            dt = h5py.string_dtype(encoding="utf-8", length=maxlen)
            meta.create_dataset("names", data=np.array(names, dtype=dt))

    print(f"[DONE] Обработано изображений: {len(names)}")
    print(f"[OUT] HDF5: {h5_path}")


if __name__ == "__main__":
    main()
