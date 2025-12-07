import os, sys, time, h5py, cv2, torch, numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

IMG_DIR = "img"
OUT_H5  = "data/depth.h5"

WEIGHTS_BEIT = "weights/dpt_beit_large_512.pt"
WEIGHTS_VIT  = "weights/dpt_large-midas-2f21e586.pt"   # ← добавь этот файл

def list_images(folder):
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")
    return sorted(os.path.join(folder,f) for f in os.listdir(folder)
                  if os.path.splitext(f.lower())[1] in exts)

def build_transform(side):
    return Compose([
        Resize(side, side, keep_aspect_ratio=True, ensure_multiple_of=32),
        NormalizeImage(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        PrepareForNet(),
    ])

def _strip_prefix(k):
    for p in ("module.","pretrained.","model.","pretrained.model.","net.","backbone."):
        if k.startswith(p): return k[len(p):]
    return k

def load_weights_tolerant(model, ckpt_path):
    print(f"[weights] загрузка: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj
    else:
        raise RuntimeError("Неизвестный формат чекпоинта.")

    model_sd = model.state_dict()
    cleaned, skipped = {}, []
    for k, v in sd.items():
        kk = _strip_prefix(k)
        if kk in model_sd and model_sd[kk].shape == v.shape:
            cleaned[kk] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    # вернём метрики совпадения, чтобы решать, «годится» ли этот вес
    matched = len(cleaned)
    need    = len(model_sd)
    ratio   = matched / max(1, need)
    print(f"[weights] matched {matched}/{need} ({ratio*100:.1f}%), skipped_in_ckpt={len(skipped)}, "
          f"missing_for_model={len(missing)}, unexpected={len(unexpected)}")
    return ratio

def make_model(backbone, device):
    # non_negative=True важно для корректных карт
    model = DPTDepthModel(backbone=backbone, non_negative=True).to(device).eval()
    return model

@torch.no_grad()
def infer_depth(model, transform, bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    sample_in = {"image": rgb}
    try:
        sample = transform(sample_in)
    except KeyError:
        # некоторые ветки midas требуют "mask"
        sample = transform({"image": rgb, "mask": np.ones((h,w), np.float32)})
    if isinstance(sample, dict):
        sample = sample.get("image", sample)
    sample = torch.from_numpy(sample).unsqueeze(0).to(device)
    pred = model(sample)
    depth = pred.squeeze().detach().cpu().numpy()
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    dmin, dmax = float(depth.min()), float(depth.max())
    depth = (depth - dmin) / (dmax - dmin + 1e-8) if dmax > dmin else np.zeros_like(depth)
    return depth.astype(np.float32)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(IMG_DIR):  raise SystemExit(f"[err] нет папки {IMG_DIR}")
    os.makedirs(os.path.dirname(OUT_H5), exist_ok=True)
    images = list_images(IMG_DIR)
    if not images: raise SystemExit(f"[err] в {IMG_DIR} нет изображений")
    if args.first: images = images[:1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device.type}" + (f" / {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    # 1) пробуем BEiT-512
    use_backbone, side, weights = "beitl16_512", 512, WEIGHTS_BEIT
    if not os.path.isfile(weights):
        print(f"[warn] {weights} не найден — сразу пробуем ViT-Large")
        use_backbone, side, weights = "vitl16_384", 384, WEIGHTS_VIT

    model = make_model(use_backbone, device)
    ratio = load_weights_tolerant(model, weights)

    # 2) если совпало мало (<70%), переключаемся на ViT-Large
    if ratio < 0.70:
        print("[warn] низкое совпадение весов — переключаюсь на ViT-Large (384). "
              "Это даст стабильные карты без артефактов.")
        model = make_model("vitl16_384", device)
        if not os.path.isfile(WEIGHTS_VIT):
            raise SystemExit("[err] нет weights/dpt_large-midas-2f21e586.pt (нужно скачать)")
        ratio = load_weights_tolerant(model, WEIGHTS_VIT)
        side = 384

    transform = build_transform(side)
    mode = "a" if args.first and os.path.exists(OUT_H5) else "w"
    with h5py.File(OUT_H5, mode) as h5:
        print(f"[io] запись в {OUT_H5} (режим {mode}), файлов: {len(images)}, input_size={side}")
        for i, p in enumerate(images, 1):
            name = os.path.basename(p)
            print(f"[{i}/{len(images)}] {name} ... ", end="", flush=True)
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                print("пропуск (не читается)"); continue
            t1 = time.time()
            depth = infer_depth(model, transform, bgr, device)
            key = "/" + name
            if key in h5: del h5[key]
            h5.create_dataset(key, data=depth, dtype="float32")
            h5.flush()
            print(f"OK ({time.time()-t1:.2f} c)")
    print(f"[done] depth.h5 готов: {OUT_H5}")

if __name__ == "__main__":
    main()
