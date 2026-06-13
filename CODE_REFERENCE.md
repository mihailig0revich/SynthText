# Справочник по коду SynthText

Этот файл описывает текущую структуру проекта, основные классы, методы и точки расширения. README отвечает на вопрос "как запустить", а этот документ помогает понять "где что живёт" и куда вносить изменения.

## Общий поток выполнения

1. `gen.py` вызывает `synthtext.cli.main()`.
2. `synthtext.cli` разбирает аргументы командной строки и собирает `GenerationConfig`.
3. `synthtext.pipeline.generate_dataset()` открывает входные `.h5`, нормализует depth/segmentation и вызывает рендерер.
4. Если включён `--ransac-stats N`, `synthtext.ransac_stats` собирает статистику отказов на первых `N` изображениях и завершает запуск без записи результата.
5. `synthtext.rendering.renderer.RendererV3.render_text()` размещает текст на подходящих областях сцены.
6. `synthtext.h5_io.H5ResultWriter` сохраняет результат в выходной HDF5.

Ключевые данные результата:

- `img`: RGB-изображение с синтетическим текстом.
- `charBB`: bounding boxes символов.
- `wordBB`: bounding boxes слов.
- `txt`: текстовые строки.
- `lang`: язык строки, если используется многоязычный источник.

## Структура Папок

Основная реализация живёт внутри пакета `synthtext/`. В корне оставлены только пользовательские точки запуска `gen.py` и `gui.py`; старые compatibility wrappers удалены.

```text
synthtext/
  cli.py, config.py, pipeline.py, h5_io.py, gui.py
  rendering/
    renderer.py, overlay.py, text_service.py, text_utils.py, colorize.py, poisson.py
  spatial/
    regions.py, geometry.py, synth_utils.py, ransac.py
  augmentation/
    noise.py, extra.py, transforms.py
  tools/
    visualize_results.py, invert_font_size.py
```

Правило навигации:

- `synthtext.pipeline` отвечает за orchestration: входные HDF5, очистку depth/seg, retries, запись результата.
- `synthtext.rendering` отвечает за текст, overlay, цвет, Poisson/colorization и финальный renderer.
- `synthtext.spatial` отвечает за depth -> XYZ, RANSAC, плоскости, homography и region placement masks.
- `synthtext.augmentation` отвечает за шумы, перспективные и дополнительные аугментации.
- `synthtext.tools` содержит вспомогательные скрипты.
- `gen.py` и `gui.py` оставлены в корне как пользовательские точки запуска. Остальные compatibility wrappers из корня удалены: импортируйте реальные модули из `synthtext.rendering`, `synthtext.spatial`, `synthtext.augmentation` и `synthtext.tools`.

## Точки входа

### `gen.py`

Тонкая обёртка над CLI:

- `main()` из `synthtext.cli` запускается при вызове `python3 gen.py`.

### `gui.py`

Тонкая обёртка над GUI:

- `main()` из `synthtext.gui` запускается при вызове `python3 gui.py`.

### `synthtext/gui.py`

Простой Tkinter-лаунчер для запуска генерации без ручного набора CLI-флагов.

- `SynthTextLauncher`
  Окно с настройками путей, числовыми параметрами, debug-флагами, preview команды, live-логом и кнопкой остановки процесса.

- `_build_command()`
  Собирает команду `python -u gen.py ...` из значений формы. Генерация всё равно проходит через обычный CLI, поэтому GUI не дублирует pipeline-логику.

- `main()`
  Создаёт окно и запускает `mainloop()`.

### `synthtext/cli.py`

Модуль командной строки.

- `build_parser() -> argparse.ArgumentParser`
  Создаёт CLI-парсер и объявляет параметры генерации: входные/выходные пути, количество изображений, число попыток, режим визуализации.

- `config_from_args(args) -> GenerationConfig`
  Преобразует результат `argparse` в объект конфигурации.

- `main(argv=None) -> None`
  Основной CLI-вход. Парсит аргументы и вызывает `generate_dataset()`.

## Конфигурация

### `synthtext/config.py`

#### `GenerationConfig`

Dataclass с настройками генерации.

Поля путей:

- `input_dir`: папка с входными `.h5`.
- `fallback_h5`: резервный `.h5`, если в `input_dir` ничего не найдено.
- `render_data_path`: папка с моделями, шрифтами и текстовыми источниками.
- `output_file`: базовый путь выходного `.h5`.
- `png_dir`: папка для PNG-вывода, если он используется рендерером.

Поля управления генерацией:

- `num_img`: сколько изображений брать из каждого входного файла; `-1` означает все.
- `instances_per_image`: сколько текстовых инстансов пытаться сгенерировать на изображение.
- `secs_per_img`: лимит времени на рендер одного изображения.
- `max_global_tries`: число повторных попыток при неудачном размещении.
- `max_h5_size_gb`: максимальный размер одного выходного `.h5` перед rollover.
- `region_workers`: число потоков для независимой проверки planarity-кандидатов в `TextRegions.filter_depth()`; `1` сохраняет последовательный режим.
- `ransac_debug`: включает подробные логи RANSAC для обычного запуска.
- `ransac_stats`: если больше `0`, включает отдельный режим статистики на первых `N` изображениях и не запускает рендер/запись HDF5.
- `placement_debug`: включает подробные логи отбора областей под размещение текста.
- `debug_progress`: печатает прогресс по файлам, изображениям и render-attempt. Автоматически включается для `--ransac-stats`, `--ransac-debug` и `--placement-debug`.
- `viz`: включает визуализацию.
- `interactive`: спрашивает путь к входной папке при старте.

## Пайплайн генерации

### `synthtext/pipeline.py`

- `make_run_id() -> str`
  Создаёт уникальный идентификатор запуска из timestamp, pid и короткого UUID. Используется в именах выходных файлов.

- `clean_depth_and_seg(depth, seg)`
  Приводит `depth` и `seg` к `float32`, заменяет `NaN/inf`, заполняет невалидные depth-значения медианой и ограничивает выбросы по 99 перцентилю.

- `generate_dataset(config: GenerationConfig) -> None`
  Главный orchestration-метод. Находит входные HDF5, создаёт `RendererV3`, открывает `H5ResultWriter`, проходит по изображениям и сохраняет успешные результаты.

- `_process_input_file(...)`
  Обрабатывает один входной `.h5`: выбирает группы image/depth/seg, приводит размеры к depth-карте, чистит данные и запускает рендер.

- `_render_with_retries(...) -> bool`
  Делает несколько попыток рендера через `RendererV3.render_text()`. Возвращает `True`, если хотя бы одна попытка дала валидный результат.

- `_should_stop_viz() -> bool`
  В интерактивной визуализации спрашивает, продолжать ли генерацию.

## HDF5 I/O

### `synthtext/h5_io.py`

- `lock_path_for(h5_path: str) -> str`
  Возвращает путь lock-файла для выходного HDF5.

- `acquire_lock_or_none(h5_path: str) -> str | None`
  Атомарно создаёт lock-файл. Возвращает путь lock-файла или `None`, если файл занят другим процессом.

- `release_lock(lock_path: str | None) -> None`
  Освобождает lock-файл.

- `make_out_path_with_index(base_path: str, run_id: str, index: int) -> str`
  Формирует имя выходного файла с `run_id` и индексом части.

- `ensure_parent_dir(path: str) -> None`
  Создаёт родительскую папку для файла.

- `list_input_h5_files(input_dir: str, fallback: str | None = None) -> list[str]`
  Возвращает отсортированный список `.h5` из папки или fallback-файл.

- `open_input_h5(h5_path: str)`
  Открывает входной HDF5 в режиме чтения и печатает диагностическую информацию.

- `pick_group(db, candidates)`
  Находит первую доступную группу из списка кандидатов. Используется для совместимости с разными схемами `.h5`.

- `read_depth_to_hw_float(depth_item)`
  Преобразует depth dataset к массиву `(H, W)` типа `float32`.

- `seg_with_attrs(seg_ds)`
  Возвращает `(seg, area, label)`. Если `area/label` нет в атрибутах, вычисляет их по `seg`.

- `add_res_to_db(imgname, res, db) -> None`
  Записывает результаты рендера в группу `/data` выходного HDF5.

#### `H5ResultWriter`

Писатель выходных HDF5 с lock-файлами и автоматическим rollover.

- `__init__(base_path, run_id, viz, max_size_gb)`
  Запоминает базовый путь, id запуска, режим визуализации и лимит размера.

- `__enter__()`
  Открывает выходной файл. В `viz=True` пишет прямо в `base_path`, иначе создаёт файл с `run_id` и индексом.

- `__exit__(exc_type, exc, tb)`
  Закрывает HDF5 и освобождает lock.

- `write(imgname, res) -> None`
  Записывает результат и при необходимости переключается на следующий файл.

- `_open_locked(path)`
  Открывает конкретный файл с lock-защитой.

- `_open_next_free(start_index)`
  Ищет следующий свободный indexed-файл.

- `_maybe_roll() -> None`
  Проверяет размер текущего `.h5` и открывает следующий файл при превышении лимита.

## Рендеринг и размещение текста

### `synthtext/rendering/renderer.py`

Это координатор рендера. В нём остались `RendererV3.__init__()`, выбор областей/текста, сбор результатов и главный `render_text()`. Низкоуровневые region/overlay helper-методы вынесены в `synthtext.spatial.regions` и `synthtext.rendering.overlay`.

Связь с `synthtext.rendering.text_utils` теперь проходит через `synthtext.rendering.text_service.TextRenderingService`. `RendererV3` не должен напрямую обращаться к `RenderFont.font_state`, `RenderFont.text_source` или `RenderFont.p_text`: для этого есть явные методы сервиса.

### `synthtext/spatial/regions.py`

Модуль поиска областей сцены и построения fronto-parallel масок.

#### `TextRegions`

Класс для поиска и фильтрации областей сцены, пригодных для размещения текста.

- `filter_rectified(mask)`
  Фильтрует выпрямленную маску области.

- `get_hw(pt, return_rot=False)`
  Оценивает размеры и ориентацию прямоугольника/набора точек.

- `filter(seg, area, label)`
  Фильтрует сегменты по площади, форме и пригодности.

- `filter_depth(xyz, seg, regions, max_planes=6)`
  Проверяет глубину и планарность кандидатов. Возвращает области, плоскости и score. Может использовать `TextRegions.region_workers > 1`, чтобы параллельно обработать независимые candidate-регионы.

- `get_regions(xyz, seg, area, label)`
  Главный метод получения кандидатов: объединяет фильтрацию сегментации и depth-проверки.

#### `get_text_placement_mask(...)`

Строит fronto-parallel маску области, куда можно помещать текст, а также матрицы перспективного перехода.

Типично используется внутри `RendererV3.filter_for_placement()`.

### `synthtext/ransac_stats.py`

Отдельный диагностический режим для поиска общих причин отказов регионов и файлов.

- `run_ransac_stats(config, input_files, limit) -> None`
  Читает первые `limit` изображений из входных `.h5`, прогоняет `TextRegions.get_regions()`, `TextRegions.filter_depth()` и проверку placement-mask. Не создаёт renderer и не пишет выходной HDF5.

- `_analyze_image(...)`
  Возвращает строку статистики по одному изображению: количество raw/shape/depth/placement регионов и статус `ok`, `no_shape_regions`, `no_depth_regions`, `no_placement_regions` или `exception`.

- `_analyze_placement(...)`
  Проверяет, сколько depth-регионов проходят `get_text_placement_mask()`, и считает причины отказов placement-этапа.

Запуск:

```bash
python3 gen.py --input-dir input --ransac-stats 100
```

В отчёте важны блоки `region events`, `placement events` и `worst images`.

Во время работы режим печатает строки `image start` и `image done`, поэтому видно, на каком ключе `.h5` сейчас находится анализ и сколько raw/shape/depth/placement регионов осталось после фильтров.

### `synthtext/rendering/overlay.py`

#### `RendererOverlayMixin`

Mixin с низкоуровневыми методами overlay-рендеринга. `RendererV3` наследуется от него, поэтому имена методов доступны как раньше через экземпляр `RendererV3`.

Группы helper-методов:

- `_compute_budgets`, `_sample_layout_text` управляют числом попыток и выбором layout/text.
- `_apply_persp_boost`, `_persp_strength_from_quad`, `_scale_edge` обслуживают перспективные helper-операции. Искусственный `overlay_persp_boost` по умолчанию выключен, чтобы финальная проекция следовала depth/RANSAC homography.
- `_sky_*` методы запрещают размещение в sky-like областях.
- `_overlay_*` методы отвечают за overlay-рендеринг, canvas, alpha, warp, outline, background rectangle и occlusion.
- `_occ_*` методы создают и применяют синтетические перекрытия.

Метод `_sample_layout_text()` оставлен в mixin как совместимая точка вызова, но фактическая очередь слов, tokenization и выбор языка находятся в `TextRenderingService`.

### `synthtext/rendering/text_service.py`

Явный bridge между сценовым рендерером и `synthtext.rendering.text_utils`.

- `TextRenderingService(data_dir)`
  Создаёт и хранит `text_utils.RenderFont`, но наружу отдаёт более узкий интерфейс для `RendererV3`.

- `sample_font() -> FontContext`
  Сэмплирует font-state, создаёт pygame/freetype font и возвращает нормализованный `aspect_ratio`.

- `estimate_layout_capacity(font_height_px, aspect_ratio, ...)`
  Оборачивает `RenderFont.get_nline_nchar()` и возвращает безопасные `nline`, `nchar`.

- `sample_layout_text(nline, nchar, min_word_len=4, max_retries=20) -> LayoutText`
  Выбирает тип текста, читает `TextSource`, токенизирует результат, хранит очередь слов и возвращает `(text, lang)`.

- `set_font_size_px(font, font_height_px)`
  Единственная точка перевода желаемой высоты текста в `font.size`.

- `render_curved(...)`, `render_sample(...)`
  Совместимые pass-through методы к `RenderFont`, если нужен низкоуровневый bitmap-render.

Практическое правило: `synthtext/rendering/renderer.py` отвечает за сцену, геометрию, placement и overlay; `synthtext/rendering/text_utils.py` отвечает за корпусы, шрифты и raster mask; `synthtext/rendering/text_service.py` описывает контракт между ними.

#### `RendererV3`

Главный класс генерации текста на изображении.

- `__init__(data_dir, max_time=None)`
  Загружает модели, источники текста, рендер шрифтов, colorizer и настройки времени.

- `filter_for_placement(xyz, seg, regions, viz=False)`
  Преобразует найденные области в маски размещения и отбрасывает неподходящие поверхности.

- `select_region_for_text(txt, font, f_layout, f_asp, place_masks, regions, ...)`
  Выбирает область под конкретный текст и шрифт.

- `place_text_textfirst(img, place_masks, regions, gap=6, ...)`
  Генерирует текстовые блоки, подбирает им размещение и собирает промежуточное представление перед цветом/наложением.

- `render_text_overlay(img, txt_str, font, selected_angle, region_coords, depth=None)`
  Рендерит текст как overlay и варпит его в выбранный quadrilateral на изображении.

- `get_num_text_regions(nregions: int) -> int`
  Решает, сколько текстовых блоков пытаться разместить для текущего числа доступных областей.

- `char2wordBB(charBB, text, ...)`
  Собирает word-level bounding boxes из character-level boxes.

- `render_text(rgb, depth, seg, area, label, ninstance=1, viz=False)`
  Основной публичный метод рендера. Возвращает список результатов с `img`, `charBB`, `wordBB`, `txt`, `lang`.

## Геометрия

### `synthtext/spatial/geometry.py`

- `warp_points(Hinv, pts_xy)`
  Применяет homography к набору `(x, y)` точек.

- `estimate_local_scale_grid(Hinv, free_mask_fp, k=9, delta=6, seed=None)`
  Оценивает локальный масштаб перспективного преобразования по сетке точек на свободной маске.

- `rescale_frontoparallel(p_fp, box_fp, p_im)`
  Масштабирует fronto-parallel координаты в соответствии с image-space координатами.

- `normalize(v, eps=1e-8)`
  Нормализует вектор с защитой от деления на ноль.

- `rot3d_scaled(n_src, n_dst, strength=1.0, max_tilt_deg=None)`
  Строит 3D-поворот между нормалями с управляемой силой наклона.

### `synthtext/spatial/ransac.py`

- `fit_plane(xyz, z_pos=None)`
  Подгоняет плоскость по 3D-точкам.

- `fit_plane_ransac(...)`
  RANSAC-подгонка плоскости с фильтрацией выбросов.

### `synthtext/spatial/synth_utils.py`

#### `LUT_RGB`

Утилита для кодирования RGB-цветов в скалярные labels и обратно через lookup table.

- `rgb2scalar(rgb)`
  Преобразует RGB-массив в scalar-label representation.

- `set_rgb_lut(myv_glyph)`
  Настраивает lookup table.

#### `DepthCamera`

Утилиты камеры и depth.

- `plane2xyz(center, ij, plane)`
  Переводит пиксельные координаты на плоскости в 3D.

- `depth2xyz(depth)`
  Переводит depth-map в 3D cloud.

- `overlay(rgb, depth)`
  Создаёт визуальное наложение RGB/depth.

Другие функции:

- `ensure_proj_z(plane_coeffs, min_z_proj)`
  Корректирует ориентацию плоскости.

- `isplanar(...)`
  Проверяет, достаточно ли область планарна.

- `get_texture_score(img, masks, labels)`
  Оценивает текстурность сегментов.

- `rot3d(v1, v2)`
  Матрица 3D-поворота между векторами.

- `unrotate2d(pts)`
  Нормализует ориентацию 2D-точек.

## Текст и шрифты

### `synthtext/rendering/text_utils.py`

- `read_lines_any_encoding(path, attempts=('utf-8', 'cp1251', 'latin-1'))`
  Читает текстовый файл с несколькими fallback-кодировками.

- `sample_weighted(p_dict)`
  Выбор ключа из словаря весов.

- `_normalize_text_lines(...)`, `_get_pil_font(...)`, `_glyph_bbox(...)`
  Внутренние helper-функции для подготовки PIL-шрифта и измерения glyph-метрик.

- `_binarize_text_mask(...)`, `_apply_text_stroke(...)`, `_resize_text_width(...)`
  Внутренние helper-функции постобработки bitmap-маски текста.

- `_load_latin1_pickle(path)`
  Загружает старые pickle-модели с `latin1` encoding.

- `move_bb(bbs, t)`
  Сдвигает bounding boxes.

- `crop_safe(arr, rect, bbs=[], pad=0)`
  Безопасно вырезает область изображения и соответствующие bounding boxes.

#### `BaselineState`

Хранит вероятностное состояние baseline-параметров текста.

- `get_sample()`
  Возвращает случайный baseline-параметр.

#### `RenderFont`

Рендерит текст в bitmap/mask и строит bounding boxes.

- `__init__(data_dir='data', ...)`
  Загружает модели символов, параметры рендера и служебные данные.

- `render_multiline(font, text)`
  Рендерит многострочный текст.

- `render_sample(font, mask)`
  Генерирует текстовый bitmap, подходящий под заданную маску.

- `get_glyph_advance(font, ch)`
  Возвращает advance символа для выбранного шрифта.

- `render_curved(font, text, ...)`
  Рендерит текст на fronto-parallel PIL canvas. Внутри разложен на измерение строк, отрисовку glyph-ов и постобработку mask/bbox.

- `get_nline_nchar(mask_size, font_height, font_width)`
  Оценивает число строк и символов, которые поместятся в маску.

- `place_text(text_arrs, back_arr, bbs)`
  Размещает отрендеренный текст на background array.

- `robust_HW(mask)`
  Оценивает устойчивые высоту/ширину маски.

- `sample_font_height_px(h_min, h_max)`
  Выбирает высоту шрифта в пикселях.

- `bb_xywh2coords(bbs)`
  Переводит bounding boxes из `(x, y, w, h)` в координаты углов.

- `visualize_bb(text_arr, bbs)`
  Отладочная визуализация boxes.

#### `FontState`

Выбирает шрифты и параметры их отображения.

- `__init__(data_dir='data')`
  Загружает список шрифтов и модели размера.

- `get_aspect_ratio(font, size=None)`
  Оценивает aspect ratio шрифта.

- `get_font_size(font, font_size_px)`
  Переводит высоту в пикселях в размер шрифта.

- `sample()`
  Сэмплирует шрифт и стиль.

- `init_font(fs)`
  Инициализирует pygame/freetype font по выбранному состоянию.

#### `BilingualTextSource`

Комбинирует несколько `TextSource` для многоязычной генерации.

- `__init__(sources: dict, p_lang=None, default_lang='en')`
  Принимает словарь `{lang: TextSource}` и вероятности языков.

- `_pick_lang()`
  Выбирает язык.

- `sample(nline_max, nchar_max, kind='WORD', return_lang=False)`
  Возвращает текст, а при `return_lang=True` ещё и выбранный язык.

#### `TextSource`

Источник текстовых строк.

- `__init__(min_nchar, fn)`
  Загружает строки из файла.

- `check_symb_frac(txt, f=0.35)`
  Проверяет долю служебных/нежелательных символов.

- `is_good(txt, f=0.35)`
  Проверяет, подходит ли строка для генерации.

- `center_align(lines)`
  Выравнивает строки по центру.

- `get_lines(nline, nword, nchar_max, f=0.35, niter=100)`
  Подбирает строки с ограничением по числу символов.

- `sample(nline_max, nchar_max, kind='WORD')`
  Главный метод сэмплинга текста.

- `sample_word(nline_max, nchar_max, niter=100)`
  Сэмплирует слова.

- `sample_line(nline_max, nchar_max)`
  Сэмплирует строки.

- `sample_para(nline_max, nchar_max)`
  Сэмплирует абзац.

## Цвет и compositing

### `synthtext/rendering/colorize.py`

#### `Layer`

Контейнер слоя.

- `__init__(alpha, color)`
  Хранит alpha-mask и RGB-цвет/изображение слоя.

#### `FontColor`

Сэмплер цветов текста на основе модели.

- `__init__(col_file)`
  Загружает модель цветов.

- `sample_normal(col_mean, col_std)`
  Сэмплирует цвет из нормального распределения.

- `sample_from_data(bg_mat)`
  Подбирает foreground/background цвета с учётом фона.

- `mean_color(arr)`
  Средний цвет массива.

- `invert(rgb)`
  Инвертирует цвет.

- `complement(rgb_color)`
  Возвращает комплементарный цвет.

- `triangle_color(col1, col2)`
  Подбирает третий цвет по цветовой схеме.

- `change_value(col_rgb, v_std=50)`
  Изменяет value/brightness цвета.

#### `Colorize`

Отвечает за цвет текста, border, shadow, blending и проверку читаемости.

- `__init__(model_dir='data')`
  Загружает цветовую модель.

- `_sample_text_color()`
  Выбирает цвет текста.

- `drop_shadow(alpha, theta, shift, size, op=0.80)`
  Генерирует слой тени.

- `border(alpha, size, kernel_type='RECT')`
  Генерирует alpha border.

- `blend(cf, cb, mode='normal')`
  Смешивает foreground/background цвета.

- `merge_two(fore, back, blend_type=None)`
  Объединяет два слоя.

- `merge_down(layers, blends=None)`
  Схлопывает список слоёв.

- `resize_im(im, osize)`
  Изменяет размер изображения.

- `occlude()`
  Генерирует параметры occlusion.

- `color_border(col_text, col_bg)`
  Выбирает цвет border.

- `color_text(text_arr, h, bg_arr)`
  Красит один текстовый bitmap.

- `process(text_arr, bg_arr, min_h)`
  Полная обработка текста цветом и эффектами.

- `check_perceptible(txt_mask, bg, txt_bg)`
  Проверяет, что текст заметен на фоне.

- `color(bg_arr, text_arr, hs, place_order=None, pad=20)`
  Красит и смешивает несколько текстовых элементов с фоном.

### `synthtext/rendering/poisson.py`

Poisson blending helper.

- `DST(x)` / `IDST(X)`
  Прямое и обратное дискретное синус-преобразование.

- `get_grads(im)`
  Градиенты изображения.

- `get_laplacian(Dx, Dy)`
  Лапласиан по градиентам.

- `poisson_solve(gx, gy, bnd)`
  Решает poisson equation для вставки.

- `blit_images(im_top, im_back, scale_grad=1.0, mode='max')`
  Вставляет верхнее изображение в фон с poisson blending.

- `contiguous_regions(mask)`
  Находит непрерывные интервалы в boolean mask.

## Аугментации и шум

### `synthtext/augmentation/extra.py`

- `bboxes_from_masks(masks)`
  Строит bounding boxes по маскам.

- `aug_rotate(img, masks, angle_deg)`
  Поворачивает изображение и маски.

- `aug_perspective(img, masks, max_jitter_ratio=0.08)`
  Перспективно искажает изображение и маски.

- `aug_gaussian_blur(img, masks=None, k=3, sigma=0.0)`
  Добавляет blur.

- `aug_brightness_contrast_gamma(...)`
  Меняет яркость, контраст и gamma.

- `aug_illumination_gradient(...)`
  Добавляет градиент освещения.

- `aug_elastic(...)`
  Делает elastic deformation.

- `aug_cutout(...)`
  Добавляет cutout-окклюзии.

- `apply_augmentations(...)`
  Применяет набор аугментаций.

### `synthtext/augmentation/noise.py`

- `noise_gaussian(img, sigma=0.02)`
  Gaussian noise.

- `noise_speckle(img, sigma=0.02)`
  Multiplicative speckle noise.

- `adjust_contrast_time_of_day(img, time_weight=None)`
  Меняет вид сцены под условное время суток.

- `noise_saltpepper(img, amount=0.008, s_vs_p=0.5)`
  Salt-and-pepper noise.

- `motion_blur(img, k=3, angle_deg=0.0)`
  Motion blur.

- `vignette(img, strength=0.1)`
  Затемнение краёв.

- `jpeg_compress_rgb(img, quality=75)`
  JPEG degradation.

- `color_jitter_rgb(...)`
  Цветовой jitter.

- `degrade_scene_rgb(...)`
  Комбинированная деградация изображения.

- `darken_scene_realistic(...)`
  Реалистичное затемнение сцены.

- `noise_bad_camera_random(img)`
  Случайная деградация плохой камерой.

- `apply_random_augmentations(...)`
  Случайный набор шумов/аугментаций.

- `apply_noise_recipe(...)`
  Применение заданного recipe.

### `synthtext/augmentation/transforms.py`

Лёгкий модуль базовых трансформаций.

- `aug_rotate(img, masks, angle_deg)`
  Поворот изображения и масок.

- `aug_perspective(img, masks, max_jitter_ratio=0.08)`
  Перспективная трансформация.

## Визуализация и debug

### `synthtext/debug_viz.py`

- `rgb_for_matplotlib(im)`
  Подготавливает изображение для matplotlib.

- `to_rgb(arr)`
  Приводит массив к RGB-виду.

- `stable_matplotlib_draw(fig=None, pause=0.25)`
  Более стабильный draw/pause для matplotlib.

- `cv2_preview(win_name, img_rgb)`
  Показывает RGB через OpenCV preview.

- `viz_textbb(fignum, text_im, bb_list, alpha=1.0)`
  Рисует bounding boxes текста.

- `viz_masks(fignum, rgb, seg, depth, label)`
  Показывает RGB, segmentation и depth.

### `synthtext/tools/visualize_results.py`

- `viz_textbb(text_im, charBB_list, wordBB, alpha=1.0)`
  Визуализирует char/word boxes.

- `main(db_fname)`
  Открывает результат генерации и показывает примеры.

## Общие утилиты

### `synthtext/common.py`

#### `Color`

Набор ANSI-кодов цветов для вывода в терминал.

#### `TimeoutException`

Исключение для операций с таймаутом.

- `colorize(num, string, bold=False, highlight=False)`
  Возвращает строку с ANSI-оформлением.

- `colorprint(colorcode, text, o=sys.stdout, bold=False)`
  Печатает цветной текст.

- `warn(msg)`
  Печатает warning.

- `error(msg)`
  Печатает error.

- `time_limit(seconds)`
  Context manager для ограничения времени выполнения блока.

## Подготовка данных и вспомогательные скрипты

### `prep_scripts/floodFill.py`

- `get_seed(sx, sy, ucm)`
  Ищет стартовую точку flood fill.

- `get_mask(ucm, viz=False)`
  Строит segmentation mask по UCM.

- `get_mask_parallel(ucm_imname)`
  Wrapper для параллельной обработки.

- `process_db_parallel(base_dir, th=0.11)`
  Читает `ucm.mat` и пишет `seg_uint16.h5`.

### `synthtext/tools/invert_font_size.py`

Скрипт построения модели соответствия font px -> pt.

### `data/newsgroup/edit.py`

- `clean_text(input_file)`
  Чистит текстовый источник.

## Практические точки расширения

- Новый CLI-флаг:
  добавить поле в `GenerationConfig`, аргумент в `build_parser()`, передачу в `config_from_args()`, затем использовать в `pipeline` или `RendererV3`.

- Новый формат входного HDF5:
  править `pick_group()`, `read_depth_to_hw_float()` или `seg_with_attrs()` в `synthtext/h5_io.py`.

- Новая стратегия размещения:
  добавлять рядом с `RendererV3.select_region_for_text()` и `RendererV3.place_text_textfirst()`.

- Новая фильтрация областей:
  начинать с `TextRegions.filter()`, `TextRegions.filter_depth()` или `RendererV3.filter_for_placement()`.

- Параллелизм:
  безопасная точка сейчас находится в `TextRegions.filter_depth()` и включается через `--region-workers N`. Не стоит шарить один `RendererV3` или один HDF5 writer между потоками: там есть mutable state, pygame/font state и файловая запись.

- Новый шум или аугментация:
  добавлять в `synthtext/augmentation/noise.py` или `synthtext/augmentation/extra.py`, а подключение делать в месте, где формируется финальный overlay/изображение.

- Новая схема логирования:
  удобнее всего начинать с `synthtext.pipeline` и `synthtext.h5_io`, затем заменить `print/colorize` в `RendererV3` на адаптер логгера.

## Что считать публичным API

Относительно стабильные точки:

- `synthtext.cli.main`
- `synthtext.config.GenerationConfig`
- `synthtext.pipeline.generate_dataset`
- `synthtext.h5_io.H5ResultWriter`
- `synthtext.rendering.renderer.RendererV3.render_text`
- `synthtext.rendering.text_utils.RenderFont`
- `synthtext.rendering.text_utils.FontState`
- `synthtext.rendering.text_utils.TextSource`
- `synthtext.rendering.colorize.Colorize`

Внутренние методы с `_` можно менять свободнее. Они описаны здесь для навигации, но не стоит завязывать на них внешний код без необходимости.
