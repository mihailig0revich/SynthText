import os, sys, h5py, cv2, numpy as np
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.color import rgb2lab
import argparse

sys.path.append(os.path.dirname(__file__))

# --- Константы ---
IMG_DIR = "img"
OUT_H5 = "data/seg.h5"

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    files = [f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in exts]
    return sorted(os.path.join(folder, f) for f in files)

def slic_labels(bgr, n_segments=600, compactness=12.0, max_iter=10):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    seg = slic(rgb, n_segments=n_segments, compactness=compactness,
               max_num_iter=max_iter, start_label=1, enforce_connectivity=True)
    return seg.astype(np.int32)

def merge_by_lab_means(bgr, labels, thresh=8.0):
    lab = rgb2lab(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype(np.float32)
    H, W = labels.shape
    K = int(labels.max())

    means = np.zeros((K+1, 3), np.float32)
    sizes = np.zeros(K+1, np.int64)
    for k in range(1, K+1):
        m = (labels == k)
        if m.any():
            means[k] = lab[m].mean(axis=0)
            sizes[k] = m.sum()

    parent = np.arange(K+1)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def uni(a,b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if sizes[rb] > 0:
            means[ra] = (means[ra]*sizes[ra] + means[rb]*sizes[rb])/(sizes[ra]+sizes[rb])
            sizes[ra] += sizes[rb]

    for y in range(H-1):
        for x in range(W-1):
            a = labels[y, x]
            b1 = labels[y+1, x]
            b2 = labels[y, x+1]
            if a != b1 and np.linalg.norm(means[a]-means[b1]) < thresh: uni(a,b1)
            if a != b2 and np.linalg.norm(means[a]-means[b2]) < thresh: uni(a,b2)

    root2new, nid = {}, 1
    out = labels.copy()
    for k in range(1, K+1):
        r = find(k)
        if r not in root2new:
            root2new[r] = nid; nid += 1
        out[labels == k] = root2new[r]

    uniq = np.unique(out)
    areas = np.array([(out == u).sum() for u in uniq], dtype=np.int64)
    return out.astype(np.uint16), areas, uniq.astype(np.int64)

def main():
    parser = argparse.ArgumentParser(description="Segmentation masks generator (SLIC + merge)")
    parser.add_argument("--first", action="store_true", help="обработать только первый файл из img/")
    args = parser.parse_args()

    if not os.path.isdir(IMG_DIR):
        raise SystemExit(f"[ERR] Не найдена папка {IMG_DIR}")
    os.makedirs(os.path.dirname(OUT_H5), exist_ok=True)

    images = list_images(IMG_DIR)
    if not images:
        raise SystemExit(f"[ERR] В {IMG_DIR} нет изображений подходящих форматов")

    if args.first:
        images = images[:1]

    # single-режим: открываем H5 в 'a', иначе в 'w'
    h5_mode = "a" if args.first and os.path.exists(OUT_H5) else "w"
    with h5py.File(OUT_H5, h5_mode) as h5:
        grp = h5["mask"] if "mask" in h5 else h5.create_group("mask")
        it = images if args.first else tqdm(images, desc="Seg")

        for p in it:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] пропуск: {p}")
                continue
            seg0 = slic_labels(bgr)
            segM, areas, labels = merge_by_lab_means(bgr, seg0)

            key = os.path.basename(p)
            if key in grp:
                del grp[key]
            dset = grp.create_dataset(key, data=segM, dtype="uint16")
            dset.attrs["area"]  = areas
            dset.attrs["label"] = labels

    print(f"✅ seg.h5 готов: {OUT_H5} ({'1 файл' if args.first else f'{len(images)} файлов'})")

if __name__ == "__main__":
    main()
