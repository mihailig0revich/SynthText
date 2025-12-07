# depth_generate_vitl.py — стабильная генерация глубины с ViT-Large (384)
# Учитывает произвольный размер входа: пред-даунскейл длинной стороны, тайлинг, мягкую склейку и постпроцесс.
import os, sys, time, h5py, cv2, torch, numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

IMG_DIR = "img"
OUT_H5  = "data/depth.h5"
WEIGHTS = "weights/dpt_large-midas-2f21e586.pt"  # ViT-Large (рекомендовано)

def list_images(folder):
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")
    return sorted(os.path.join(folder,f) for f in os.listdir(folder)
                  if os.path.splitext(f.lower())[1] in exts)

def build_transform_384():
    return Compose([
        Resize(384, 384, keep_aspect_ratio=True, ensure_multiple_of=32),
        NormalizeImage(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        PrepareForNet(),
    ])

def strict_load(model, path):
    print(f"[weights] загрузка: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError("Ожидался dict с state_dict")
    sd = obj.get("state_dict", obj.get("model", obj))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model_sd = model.state_dict()
    matched = sum(1 for k,v in sd.items() if k in model_sd and model_sd[k].shape == v.shape)
    need = len(model_sd)
    ratio = matched / max(1, need)
    print(f"[weights] matched {matched}/{need} ({ratio*100:.1f}%), missing={len(missing)}, unexpected={len(unexpected)}")
    return ratio

@torch.no_grad()
def infer_single(model, transform, bgr, device):
    """Инференс на одном кадре (без тайлинга)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # некоторые версии midas.transforms хотят mask — подадим при необходимости
    sample_in = {"image": rgb}
    try:
        sample = transform(sample_in)
    except KeyError:
        sample = transform({"image": rgb, "mask": np.ones((h,w), np.float32)})
    if isinstance(sample, dict):
        sample = sample.get("image", sample)

    x = torch.from_numpy(sample).unsqueeze(0).to(device)
    y = model(x).squeeze().detach().cpu().numpy()
    y = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)
    dmin, dmax = float(y.min()), float(y.max())
    y = (y - dmin) / (dmax - dmin + 1e-8) if dmax > dmin else np.zeros_like(y)
    return y.astype(np.float32)

def hann2d(h, w, edge):
    """2D косиновое перо для плавной склейки: flat в центре, спад на краях шириной edge."""
    y = np.hanning(edge*2+1)
    x = np.hanning(edge*2+1)
    wy = np.ones(h); wx = np.ones(w)
    wy[:edge] = y[:edge]; wy[-edge:] = y[-edge:]
    wx[:edge] = x[:edge]; wx[-edge:] = x[-edge:]
    return wy[:,None] * wx[None,:]

@torch.no_grad()
def infer_tiled(model, transform, bgr, device, tile=384, overlap=64):
    """Тайлинговый инференс с перьевой склейкой."""
    H, W = bgr.shape[:2]
    step = tile - overlap*2
    if step <= 0:
        raise ValueError("overlap слишком большой: нужно tile > 2*overlap")

    out = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    feather = hann2d(tile, tile, overlap).astype(np.float32)

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            # если у правого/нижнего края тайл меньше, расширим окно на недостающие пиксели
            yy0 = max(0, y1 - tile); xx0 = max(0, x1 - tile)
            patch = bgr[yy0:y1, xx0:x1]
            if patch.shape[0] < 16 or patch.shape[1] < 16:
                continue

            d = infer_single(model, transform, patch, device)
            # нормализуем внутри тайла (стабильнее при разном освещении)
            dmin, dmax = float(d.min()), float(d.max())
            if dmax > dmin: d = (d - dmin) / (dmax - dmin)
            fh, fw = feather[:d.shape[0], :d.shape[1]],  # обрезаем перо если край
            fh = feather[:d.shape[0], :d.shape[1]]

            out[yy0:y1, xx0:x1] += d * fh
            wsum[yy0:y1, xx0:x1] += fh

    out = np.divide(out, np.maximum(wsum, 1e-6)).astype(np.float32)
    return out

def pre_downscale_long_side(img, max_side):
    if max_side <= 0:  # без пред-ресайза
        return img, 1.0
    H, W = img.shape[:2]
    s = max(H, W)
    if s <= max_side:
        return img, 1.0
    scale = max_side / float(s)
    newW, newH = int(round(W*scale)), int(round(H*scale))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
    return resized, 1.0/scale  # обратный масштаб (для информации, тут не нужен)

def postprocess_klex(depth, use_guided=False, bgr=None):
    """Лёгкое подавление «клякс», с сохранением границ."""
    d8 = (depth * 255).astype(np.uint8)
    d8 = cv2.medianBlur(d8, 5)
    d8 = cv2.bilateralFilter(d8, d=7, sigmaColor=25, sigmaSpace=7)
    if use_guided and bgr is not None:
        try:
            import cv2.ximgproc as xip
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            guide = cv2.normalize(gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            d8 = xip.guidedFilter(guide, d8, radius=8, eps=1e-3)
        except Exception:
            pass
    return d8.astype(np.float32)/255.0

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", action="store_true", help="только первый файл (быстрый тест)")
    ap.add_argument("--pre_max_side", type=int, default=0,
                    help="пред-даунскейл длинной стороны до этого размера (0=выкл). Рекомендуется 1024..1536 для 4K")
    ap.add_argument("--tile", type=int, default=0,
                    help="размер тайла (0=без тайлинга). Для панорам используйте 384")
    ap.add_argument("--overlap", type=int, default=64,
                    help="перекрытие тайлов по краям (обычно 64)")
    ap.add_argument("--smooth", action="store_true",
                    help="включить лёгкий постпроцессинг (median+bilateral/guided)")
    args = ap.parse_args()

    if not os.path.isdir(IMG_DIR):  raise SystemExit(f"[err] нет папки {IMG_DIR}")
    if not os.path.isfile(WEIGHTS): raise SystemExit(f"[err] нет файла {WEIGHTS}")
    os.makedirs(os.path.dirname(OUT_H5), exist_ok=True)

    images = list_images(IMG_DIR)
    if not images: raise SystemExit(f"[err] в {IMG_DIR} нет изображений")
    if args.first: images = images[:1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device.type}" + (f" / {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    model = DPTDepthModel(backbone="vitl16_384", non_negative=True).to(device).eval()
    ratio = strict_load(model, WEIGHTS)
    if ratio < 0.90:
        print("[warn] Совпадение весов ниже 90%. Проверь версию midas/ и файл весов — качество может страдать.")

    transform = build_transform_384()

    mode = "a" if args.first and os.path.exists(OUT_H5) else "w"
    with h5py.File(OUT_H5, mode) as h5:
        print(f"[io] запись в {OUT_H5} (режим {mode}), файлов: {len(images)}, "
              f"pre_max_side={args.pre_max_side}, tile={args.tile}, overlap={args.overlap}")
        for i, p in enumerate(images, 1):
            name = os.path.basename(p)
            print(f"[{i}/{len(images)}] {name} ... ", end="", flush=True)
            orig = cv2.imread(p, cv2.IMREAD_COLOR)
            if orig is None:
                print("пропуск (не читается)"); continue

            # пред-даунскейл длинной стороны (для 4K, панорам и т.п.)
            img, _ = pre_downscale_long_side(orig, args.pre_max_side)

            t0 = time.time()
            if args.tile and args.tile > 0:
                depth = infer_tiled(model, transform, img, device, tile=args.tile, overlap=args.overlap)
                # ресайз обратно к размеру исходника
                depth = cv2.resize(depth, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                depth = infer_single(model, transform, img, device)
                depth = cv2.resize(depth, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC)

            if args.smooth:
                depth = postprocess_klex(depth, use_guided=False, bgr=orig)

            key = "/" + name
            if key in h5: del h5[key]
            h5.create_dataset(key, data=depth, dtype="float32")
            h5.flush()
            print(f"OK ({time.time()-t0:.2f} c)")

    print(f"[done] depth.h5 готов: {OUT_H5}")

if __name__ == "__main__":
    main()
