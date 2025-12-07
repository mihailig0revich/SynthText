# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import matplotlib
try:
    matplotlib.use("TkAgg")  # на Windows обычно ок
except Exception:
    pass

import matplotlib.pyplot as plt
plt.ion()
print("[VIZ] matplotlib backend:", matplotlib.get_backend())

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
from common import TimeoutException
from PIL import Image

try:
    RESAMPLE = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except Exception:
    RESAMPLE = getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', Image.BICUBIC))
    RESAMPLE_NEAREST = getattr(Image, 'NEAREST', Image.BILINEAR)

# ------------------------- Configuration -------------------------
NUM_IMG = -1
INSTANCE_PER_IMAGE = 1
SECS_PER_IMG = 5

# Путь к HDF5-файлу с бэкграундами
DB_FNAME = r"C:\code\SynthText-python3\street\bg_data\bg_data.h5"

# Папка с ресурсами для текста (корпус + шрифты)
RENDER_DATA_PATH = r"C:\code\SynthText-python3\data"

# Базовый путь для выходного H5
OUT_FILE = 'results/SynthText.h5'

MAX_GLOBAL_TRIES = 8

# NEW: лимит размера одного выходного H5-файла (в гигабайтах)
MAX_H5_SIZE_GB = 15.0
# -----------------------------------------------------------------


def clean_depth_and_seg(depth, seg):
    # float32
    depth = depth.astype(np.float32, copy=False)
    seg   = seg.astype(np.float32,   copy=False)

    # заменяем NaN/Inf
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    seg   = np.nan_to_num(seg,   nan=0.0, posinf=0.0, neginf=0.0)

    # поднимаем невалидную/нулевую глубину к медиане положительных
    if np.any(depth > 0):
        med = float(np.median(depth[depth > 0]))
        if med <= 0:
            med = 1.0
        depth[depth <= 0] = med
    else:
        depth[...] = 1.0

    # ограничим очень большие значения глубины
    hi = float(np.percentile(depth, 99))
    if hi > 0:
        depth = np.clip(depth, 1.0, hi)

    return depth, seg


def get_data():
    """Открывает локальный dset.h5"""
    if not osp.exists(DB_FNAME):
        raise FileNotFoundError(f"Не найден {DB_FNAME}")
    db = h5py.File(DB_FNAME, 'r')
    print("[H5] top-level keys:", list(db.keys()))
    return db


def pick_group(db, candidates):
    """Возвращает (group, выбранное_имя) из списка возможных имён."""
    for name in candidates:
        if name in db:
            return db[name], name
    raise KeyError(f"Не нашёл ни одну из групп {candidates}. Доступны: {list(db.keys())}")


def add_res_to_db(imgname, res, db):
    """
    Сохраняет синтетические результаты в выходной H5 в формате
    совместимом с оригинальным SynthText:

    /data/<imgname>_i  (dataset с изображением)
        attrs['charBB'], attrs['wordBB'], attrs['txt']
    """
    for i, r in enumerate(res):
        dname = f"{imgname}_{i}"
        dset = db['data'].create_dataset(dname, data=r['img'])
        dset.attrs['charBB'] = r['charBB']
        dset.attrs['wordBB'] = r['wordBB']
        L = [t.encode("ascii", "ignore") for t in r['txt']]
        dset.attrs['txt'] = L


def _read_depth_to_hw_float(depth_item):
    """Приводит depth к (H, W) float32."""
    d = np.array(depth_item[:])
    if d.ndim == 2:
        return d.astype(np.float32)
    elif d.ndim == 3:
        if d.shape[0] in (1, 2, 3) and d.shape[0] != d.shape[-1]:
            d = np.moveaxis(d, 0, -1)
        if d.shape[2] == 1:
            return d[..., 0].astype(np.float32)
        elif d.shape[2] >= 3:
            return d[..., 1].astype(np.float32)
        else:
            return d.mean(axis=2).astype(np.float32)
    else:
        raise ValueError(f"Неожиданная форма depth: {d.shape}")


def _seg_with_attrs(seg_ds):
    """Возвращает (seg_float32, area, label)."""
    seg = np.array(seg_ds[:]).astype('float32')
    if 'area' in seg_ds.attrs and 'label' in seg_ds.attrs:
        area = seg_ds.attrs['area']
        label = seg_ds.attrs['label']
    else:
        labels, counts = np.unique(seg.astype(np.int32), return_counts=True)
        area = counts.astype(np.float32)
        label = labels.astype(np.int32)
    return seg, area, label


def _assert_render_assets(path):
    """Проверяет наличие корпуса и шрифтов."""
    need = [
        osp.join(path, 'newsgroup', 'newsgroup.txt'),
        osp.join(path, 'fonts'),
    ]
    missing = [p for p in need if not osp.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"❌ Не найдены ресурсы для рендеринга текста!\n"
            f"Ожидаю:\n  {need[0]}\n  {need[1]}"
        )


# ================== NEW: работа с выходными H5-файлами ==================

def _make_out_path_with_index(base_path: str, index: int) -> str:
    """
    base_path = 'results/SynthText.h5'
    index = 0   -> 'results/SynthText.h5'
    index = 1   -> 'results/SynthText_0001.h5'
    index = 2   -> 'results/SynthText_0002.h5'
    и т.д.
    """
    if index == 0:
        return base_path
    root, ext = osp.splitext(base_path)
    return f"{root}_{index:04d}{ext}"


def _open_out_h5(base_path: str, index: int):
    """
    Открывает новый выходной H5-файл и создаёт группу /data.
    Возвращает (h5_file, path).
    """
    path = _make_out_path_with_index(base_path, index)
    os.makedirs(osp.dirname(path), exist_ok=True)
    f = h5py.File(path, 'w')
    f.create_group('/data')
    print(colorize(Color.GREEN, f"[H5] Opened output file: {path}", bold=True))
    return f, path


def _maybe_roll_output_h5(out_db, out_path, base_path, index, max_gb: float):
    """
    Проверяет размер текущего H5-файла.
    Если размер >= max_gb, закрывает его и открывает новый с индексом +1.
    Возвращает (out_db, out_path, new_index).
    """
    try:
        out_db.flush()
    except Exception:
        pass

    try:
        if osp.exists(out_path):
            size_bytes = os.path.getsize(out_path)
        else:
            size_bytes = 0
    except Exception:
        size_bytes = 0

    if size_bytes >= max_gb * (1024 ** 3):
        # закрываем текущий и открываем новый
        try:
            out_db.close()
        except Exception:
            pass
        new_index = index + 1
        out_db, out_path = _open_out_h5(base_path, new_index)
        return out_db, out_path, new_index
    else:
        return out_db, out_path, index

# =======================================================================


def main(viz=False):
    print(colorize(Color.BLUE, 'getting data..', bold=True))
    db = get_data()
    print(colorize(Color.BLUE, '\t-> done', bold=True))

    # Проверим наличие корпуса и шрифтов
    _assert_render_assets(RENDER_DATA_PATH)

    # Группы с изображениями / глубиной / сегментацией
    img_g, img_g_name     = pick_group(db, ['images', 'image', 'img'])
    depth_g, depth_g_name = pick_group(db, ['depth', 'depths'])
    seg_g, seg_g_name     = pick_group(db, ['seg', 'segs', 'mask'])

    common_keys = sorted(set(img_g.keys()) & set(depth_g.keys()) & set(seg_g.keys()))
    if not common_keys:
        raise RuntimeError("Нет общих ключей между группами!")

    # Ограничим количество по NUM_IMG
    if NUM_IMG > 0:
        keys = common_keys[:NUM_IMG]
    else:
        keys = common_keys

    total_bg = len(keys)

    print(f"[H5] using groups: image='{img_g_name}', depth='{depth_g_name}', seg='{seg_g_name}'")
    print(colorize(Color.GREEN, 'Storing the output in: ' + OUT_FILE, bold=True))

    # NEW: открываем первый выходной H5 и будем роллить по мере роста
    out_index = 0
    out_db, out_path = _open_out_h5(OUT_FILE, out_index)

    # теперь передаем путь к data/, где лежит newsgroup и fonts
    RV3 = RendererV3(RENDER_DATA_PATH, max_time=SECS_PER_IMG)

    # NEW: счётчик всех сгенерированных картинок
    total_generated = 0

    for i, imname in enumerate(keys):
        try:
            img_np = np.array(img_g[imname][:])
            img_pil = Image.fromarray(img_np)

            depth = _read_depth_to_hw_float(depth_g[imname])
            seg, area, label = _seg_with_attrs(seg_g[imname])

            sz = depth.shape[:2][::-1]
            img = np.array(img_pil.resize(sz, RESAMPLE))
            seg = np.array(Image.fromarray(seg).resize(sz, RESAMPLE_NEAREST))
            depth, seg = clean_depth_and_seg(depth, seg)

            # Индекс фона
            print(colorize(Color.RED, f'{i+1}/{total_bg}', bold=True))

            saved_any = False
            new_generated_for_this_bg = 0

            for attempt in range(1, MAX_GLOBAL_TRIES + 1):
                res = RV3.render_text(
                    img, depth, seg, area, label,
                    ninstance=INSTANCE_PER_IMAGE, viz=viz
                )

                # res — список инстансов; каждый инстанс — словарь с 'img','txt','charBB','wordBB'
                if res and len(res) > 0 and isinstance(res[0].get('img', []), np.ndarray):
                    # Сохраняем в текущий H5
                    add_res_to_db(imname, res, out_db)
                    new_generated_for_this_bg = len(res)
                    total_generated += new_generated_for_this_bg
                    saved_any = True

                    # NEW: после записи — проверяем, не пора ли открыть новый файл
                    out_db, out_path, out_index = _maybe_roll_output_h5(
                        out_db, out_path, OUT_FILE, out_index, MAX_H5_SIZE_GB
                    )

                    print(colorize(
                        Color.GREEN,
                        f"[OK] image '{imname}' (attempt {attempt}) -> "
                        f"{new_generated_for_this_bg} synthetic",
                        bold=True
                    ))
                    break
                else:
                    print(colorize(
                        Color.YELLOW,
                        f"[WARN] attempt {attempt}: no placement, retrying...",
                        bold=True
                    ))

            if not saved_any:
                print(colorize(Color.RED, "[FAIL] all attempts failed for this image", bold=True))

            # NEW: прогресс по фонам и общему числу сгенерированных картинок
            progress = 100.0 * (i + 1) / max(1, total_bg)
            print(colorize(
                Color.CYAN,
                f"[PROGRESS] {i+1}/{total_bg} backgrounds "
                f"({progress:5.1f}%), total synthetic images: {total_generated}",
                bold=True
            ))

            if viz:
                # старый режим с паузой
                ans = input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True))
                if 'q' in ans:
                    break

        except Exception:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue

    db.close()
    try:
        out_db.close()
    except Exception:
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    args = parser.parse_args()
    main(args.viz)
