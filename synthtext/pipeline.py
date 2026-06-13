import os
import re
import time
import traceback
import uuid

import numpy as np
from PIL import Image

from .common import Color, colorize

from .config import GenerationConfig
from .h5_io import (
    H5ResultWriter,
    list_input_h5_files,
    open_input_h5,
    pick_group,
    read_depth_to_hw_float,
    seg_with_attrs,
)

try:
    RESAMPLE = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except Exception:
    RESAMPLE = getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", Image.BICUBIC))
    RESAMPLE_NEAREST = getattr(Image, "NEAREST", Image.BILINEAR)


def make_run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    rnd = uuid.uuid4().hex[:6]
    return re.sub(r"[^0-9A-Za-z_]+", "_", f"{ts}_{pid}_{rnd}")


def clean_depth_and_seg(depth, seg):
    depth = depth.astype(np.float32, copy=False)
    seg = seg.astype(np.float32, copy=False)

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

    valid = depth > 0
    if np.nanmax(depth) >= 65535.0:
        valid &= depth < 65535.0

    if np.any(valid):
        med = float(np.median(depth[valid]))
        if med <= 0:
            med = 1.0
        if seg.shape[:2] == depth.shape[:2]:
            seg[~valid] = 0
        depth[~valid] = med
    else:
        depth[...] = 1.0

    hi = float(np.percentile(depth[depth > 0], 99))
    if hi > 0:
        depth = np.clip(depth, 1.0, hi)

    if hi > 1000.0:
        depth = depth / 1000.0

    return depth, seg


def _resolve_input_dir(config: GenerationConfig) -> str:
    if not config.interactive:
        return config.input_dir

    try:
        user_dir = input(
            f"\nВведите путь к ПАПКЕ, из которой читать .h5-файлы\n"
            f"[Enter = использовать по умолчанию: {config.input_dir}]\n> "
        ).strip()
    except EOFError:
        user_dir = ""
    return user_dir if user_dir else config.input_dir


def generate_dataset(config: GenerationConfig) -> None:
    from .spatial.regions import TextRegions

    TextRegions.region_workers = max(1, int(config.region_workers))
    TextRegions.ransac_debug = bool(config.ransac_debug)

    selected_input_dir = _resolve_input_dir(config)
    input_files = list_input_h5_files(
        input_dir=selected_input_dir,
        fallback=config.fallback_h5,
    )

    print(colorize(Color.BLUE, f"\nНашёл {len(input_files)} .h5-файлов для обработки:", bold=True))
    print("   Папка ввода:", selected_input_dir)
    for path in input_files:
        print("   ", path)

    if int(config.ransac_stats) > 0:
        from .ransac_stats import run_ransac_stats

        run_ransac_stats(config, input_files, int(config.ransac_stats))
        return

    output_dir = os.path.dirname(config.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.png_dir, exist_ok=True)

    run_id = make_run_id()
    print(colorize(Color.BLUE, f"[RUN] id = {run_id}", bold=True))
    if TextRegions.region_workers > 1:
        print(colorize(Color.BLUE, f"[REGIONS] plane fitting workers = {TextRegions.region_workers}", bold=True))
    if TextRegions.ransac_debug:
        print(colorize(Color.BLUE, "[RANSAC] debug logging enabled", bold=True))
    if config.placement_debug:
        print(colorize(Color.BLUE, "[PLACEMENT] debug logging enabled", bold=True))
    if config.debug_progress:
        print(colorize(Color.BLUE, "[PROGRESS] debug progress enabled", bold=True))

    if config.viz:
        print(colorize(Color.GREEN, "Storing the output in: " + config.output_file, bold=True))

    with H5ResultWriter(
        base_path=config.output_file,
        run_id=run_id,
        viz=config.viz,
        max_size_gb=config.max_h5_size_gb,
    ) as writer:
        from .rendering.renderer import RendererV3

        renderer = RendererV3(config.render_data_path, max_time=config.secs_per_img)
        renderer.debug_regions = bool(config.placement_debug)
        renderer.debug_txt = bool(config.placement_debug)
        global_idx = 0

        for file_idx, h5_path in enumerate(input_files):
            print(colorize(Color.MAGENTA, f"[INPUT] {file_idx + 1}/{len(input_files)}: {h5_path}", bold=True))

            try:
                db = open_input_h5(h5_path)
            except Exception:
                traceback.print_exc()
                print(colorize(Color.RED, "[H5] не удалось открыть файл, пропускаю", bold=True))
                continue

            try:
                processed_count, stop_requested = _process_input_file(
                    db,
                    file_idx,
                    h5_path,
                    input_files,
                    config,
                    renderer,
                    writer,
                    global_idx,
                )
                global_idx += processed_count
                if stop_requested:
                    return
            finally:
                try:
                    db.close()
                except Exception:
                    pass


def _process_input_file(db, file_idx, h5_path, input_files, config, renderer, writer, global_idx_start: int):
    try:
        img_g, img_g_name = pick_group(db, ["images", "image", "img"])
        depth_g, depth_g_name = pick_group(db, ["depth", "depths"])
        seg_g, seg_g_name = pick_group(db, ["seg", "segs", "mask"])

        common = sorted(set(img_g.keys()) & set(depth_g.keys()) & set(seg_g.keys()))
        if not common:
            print(colorize(Color.YELLOW, "[H5] Нет общих ключей между группами image/depth/seg, пропускаю файл", bold=True))
            return 0, False

        print(f"[H5] using groups: image='{img_g_name}', depth='{depth_g_name}', seg='{seg_g_name}'")
    except Exception:
        traceback.print_exc()
        print(colorize(Color.RED, "[H5] Ошибка при разборе групп, пропускаю файл", bold=True))
        return 0, False

    n_in_file = len(common) if config.num_img <= 0 else min(config.num_img, len(common))
    global_idx = int(global_idx_start)
    processed_count = 0
    if config.debug_progress:
        print(
            f"[PROGRESS] file {file_idx + 1}/{len(input_files)} "
            f"images={n_in_file} path={h5_path}",
            flush=True,
        )

    for i, imname in enumerate(common[:n_in_file]):
        global_idx += 1
        processed_count += 1
        try:
            if config.debug_progress:
                print(
                    f"[PROGRESS] image start global={global_idx} "
                    f"file={file_idx + 1}/{len(input_files)} "
                    f"image={i + 1}/{n_in_file} key={imname}",
                    flush=True,
                )

            img_np = np.array(img_g[imname][:])
            img_pil = Image.fromarray(img_np)
            depth = read_depth_to_hw_float(depth_g[imname])
            seg, area, label = seg_with_attrs(seg_g[imname])

            size = depth.shape[:2][::-1]
            img = np.array(img_pil.resize(size, RESAMPLE))
            seg = np.array(Image.fromarray(seg).resize(size, RESAMPLE_NEAREST))
            depth, seg = clean_depth_and_seg(depth, seg)

            print(colorize(
                Color.RED,
                f"{global_idx} (file {file_idx + 1}/{len(input_files)}, img {i + 1}/{n_in_file})",
                bold=True,
            ))

            saved_any = _render_with_retries(renderer, writer, imname, img, depth, seg, area, label, config)
            if config.debug_progress:
                status = "ok" if saved_any else "failed"
                print(
                    f"[PROGRESS] image done global={global_idx} "
                    f"image={i + 1}/{n_in_file} status={status}",
                    flush=True,
                )
            if not saved_any:
                print(colorize(Color.RED, "[FAIL] all attempts failed for this image", bold=True))

            if config.viz and _should_stop_viz():
                return processed_count, True
        except Exception:
            traceback.print_exc()
            print(colorize(Color.GREEN, ">>>> CONTINUING....", bold=True))

    return processed_count, False


def _render_with_retries(renderer, writer, imname, img, depth, seg, area, label, config: GenerationConfig) -> bool:
    for attempt in range(1, config.max_global_tries + 1):
        if config.debug_progress:
            print(
                f"[PROGRESS] render attempt {attempt}/{config.max_global_tries} key={imname}",
                flush=True,
            )
        res = renderer.render_text(
            img,
            depth,
            seg,
            area,
            label,
            ninstance=config.instances_per_image,
            viz=config.viz,
        )

        if res and len(res) > 0 and isinstance(res[0].get("img", []), np.ndarray):
            writer.write(imname, res)
            if config.viz:
                _show_viz_result(res, img, seg, depth)
            print(colorize(
                Color.GREEN,
                f"[OK] saved {len(res)} instance(s) for '{imname}' into H5 (attempt {attempt})",
                bold=True,
            ))
            return True

        print(colorize(Color.YELLOW, f"[WARN] attempt {attempt}: no placement, retrying...", bold=True))

    return False


def _show_viz_result(res, original_img, seg, depth) -> None:
    try:
        from .debug_viz import viz_generation_overview, viz_textbb

        item = next((r for r in res if isinstance(r.get("img", None), np.ndarray)), None)
        if item is None:
            return

        word_bb = item.get("wordBB", None)
        char_bb = item.get("charBB", None)
        show_bb = word_bb if word_bb is not None and hasattr(word_bb, "shape") and word_bb.size > 0 else char_bb
        bb_list = [show_bb] if show_bb is not None and hasattr(show_bb, "shape") and show_bb.size > 0 else []
        viz_textbb(1, item["img"], bb_list, alpha=0.8)
        viz_generation_overview(2, original_img, seg, depth, item["img"])
    except Exception as exc:
        print("[VIZ] failed to display generated image:", repr(exc))


def _should_stop_viz() -> bool:
    print(colorize(Color.RED, "continue? (press Continue in GUI or enter; q to exit): ", True), end="", flush=True)
    ans = input()
    return "q" in ans
