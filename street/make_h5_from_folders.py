import h5py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Путь к основной папке
base_dir = Path(r"C:\code\SynthText-python3\street\bg_data")

# Подпапки
folders = {
    "images": base_dir / "img",
    "depth": base_dir / "depth",
    "seg": base_dir / "seg",
}

# Проверяем, что все три папки существуют
for k, p in folders.items():
    if not p.exists():
        raise FileNotFoundError(f"Папка не найдена: {p}")

# Файл для записи
out_path = base_dir / "dset.h5"

# Открываем файл на запись
with h5py.File(out_path, "w") as h5f:
    print(f"[INFO] Создаётся HDF5-файл: {out_path}")

    for name, folder in folders.items():
        print(f"[INFO] Обработка папки: {name}")
        group = h5f.create_group(name)

        files = sorted(folder.glob("*.*"))
        for i, path in enumerate(tqdm(files, desc=name)):
            try:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[WARN] Не удалось прочитать {path}")
                    continue

                # Конвертация 16-битных изображений в 8-битные при необходимости
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                # Сохраняем без сжатия
                group.create_dataset(
                    name=f"{i:05d}",
                    data=img,
                )
            except Exception as e:
                print(f"[ERROR] {path}: {e}")

print(f"✅ Готово! HDF5 сохранён в {out_path}")
