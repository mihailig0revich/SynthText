import os, glob

# Папка с изображениями (относительно корня проекта)
image_dir = "street/img"

# Путь, куда сохранить список
output_list = "data/list.txt"

# Собираем все изображения
extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(image_dir, ext)))

# Преобразуем пути к относительным (от корня проекта)
files = [os.path.relpath(path, start=".") for path in sorted(files)]

# Сохраняем
with open(output_list, "w", encoding="utf-8") as f:
    for path in files:
        f.write(path + "\n")

print(f"✅ Список сохранён: {output_list}")
print(f"Добавлено файлов: {len(files)}")
