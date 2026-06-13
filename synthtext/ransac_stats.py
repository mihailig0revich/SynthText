"""Batch diagnostics for region/RANSAC/placement filtering."""

from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from .spatial import synth_utils as su

from .h5_io import open_input_h5, pick_group, read_depth_to_hw_float, seg_with_attrs
from .pipeline import RESAMPLE, RESAMPLE_NEAREST, clean_depth_and_seg
from .spatial.regions import TextRegions, get_text_placement_mask


def run_ransac_stats(config, input_files, limit: int) -> None:
    limit = int(limit)
    if limit <= 0:
        return

    old_collector = getattr(TextRegions, "stats_collector", None)
    old_debug = bool(getattr(TextRegions, "ransac_debug", False))
    old_verbose = bool(getattr(TextRegions, "verbose", False))
    old_workers = int(getattr(TextRegions, "region_workers", 1) or 1)

    totals = Counter()
    region_events = Counter()
    placement_events = Counter()
    file_totals = defaultdict(Counter)
    rows = []
    progress = bool(getattr(config, "debug_progress", False))
    placement_geometry = {
        "persp_strength": float(getattr(config, "persp_strength", 1.0)),
        "persp_max_tilt_deg": getattr(config, "persp_max_tilt_deg", None),
    }

    processed = 0
    try:
        TextRegions.ransac_debug = False
        TextRegions.verbose = False
        TextRegions.region_workers = 1
        if progress:
            print(f"[RANSAC-STATS] progress enabled: target={limit}, files={len(input_files)}", flush=True)

        for file_idx, h5_path in enumerate(input_files):
            if processed >= limit:
                break
            if progress:
                print(f"[RANSAC-STATS] file {file_idx + 1}/{len(input_files)}: {h5_path}", flush=True)

            try:
                db = open_input_h5(h5_path)
            except Exception as exc:
                totals["file_open_failed"] += 1
                file_totals[str(h5_path)]["file_open_failed"] += 1
                print(f"[STATS] cannot open {h5_path}: {repr(exc)}")
                continue

            try:
                try:
                    img_g, _ = pick_group(db, ["images", "image", "img"])
                    depth_g, _ = pick_group(db, ["depth", "depths"])
                    seg_g, _ = pick_group(db, ["seg", "segs", "mask"])
                    common = sorted(set(img_g.keys()) & set(depth_g.keys()) & set(seg_g.keys()))
                except Exception as exc:
                    totals["file_parse_failed"] += 1
                    file_totals[str(h5_path)]["file_parse_failed"] += 1
                    print(f"[STATS] cannot parse groups in {h5_path}: {repr(exc)}")
                    continue

                if not common:
                    totals["file_no_common_keys"] += 1
                    file_totals[str(h5_path)]["file_no_common_keys"] += 1
                    print(f"[STATS] no common image/depth/seg keys in {h5_path}")
                    continue

                for local_idx, imname in enumerate(common):
                    if processed >= limit:
                        break
                    processed += 1
                    if progress:
                        print(
                            f"[RANSAC-STATS] image start {processed}/{limit} "
                            f"file={file_idx + 1}/{len(input_files)} "
                            f"image={local_idx + 1}/{len(common)} key={imname}",
                            flush=True,
                        )
                    row = _analyze_image(img_g, depth_g, seg_g, imname, placement_geometry)
                    row["file"] = str(h5_path)
                    row["file_idx"] = file_idx + 1
                    rows.append(row)
                    if progress:
                        print(
                            f"[RANSAC-STATS] image done {processed}/{limit} "
                            f"status={row['status']} raw={row['raw_regions']} "
                            f"shape={row['shape_regions']} depth={row['depth_regions']} "
                            f"placement={row['placement_regions']}",
                            flush=True,
                        )

                    totals["images"] += 1
                    totals[row["status"]] += 1
                    file_totals[str(h5_path)]["images"] += 1
                    file_totals[str(h5_path)][row["status"]] += 1
                    region_events.update(row["region_events"])
                    placement_events.update(row["placement_events"])
            finally:
                try:
                    db.close()
                except Exception:
                    pass
    finally:
        TextRegions.stats_collector = old_collector
        TextRegions.ransac_debug = old_debug
        TextRegions.verbose = old_verbose
        TextRegions.region_workers = old_workers

    _print_summary(totals, file_totals, region_events, placement_events, rows)


def _analyze_image(img_g, depth_g, seg_g, imname, placement_geometry=None):
    events = []
    old_collector = getattr(TextRegions, "stats_collector", None)
    TextRegions.stats_collector = events

    try:
        img_np = np.array(img_g[imname][:])
        img_pil = Image.fromarray(img_np)
        depth = read_depth_to_hw_float(depth_g[imname])
        seg, area, label = seg_with_attrs(seg_g[imname])

        size = depth.shape[:2][::-1]
        img = np.array(img_pil.resize(size, RESAMPLE))
        seg = np.array(Image.fromarray(seg).resize(size, RESAMPLE_NEAREST))
        depth, seg = clean_depth_and_seg(depth, seg)

        xyz = su.DepthCamera.depth2xyz(depth)

        labels, counts = np.unique(seg.astype(np.int32), return_counts=True)
        raw_regions = int(np.sum(labels != 0))
        raw_pixels = int(np.sum(counts[labels != 0]))

        shape_regions = TextRegions.get_regions(xyz, seg.astype(np.int32), area, label)
        shape_count = int(len(shape_regions.get("label", [])))

        depth_regions = TextRegions.filter_depth(xyz, seg.astype(np.int32), shape_regions)
        depth_count = int(len(depth_regions.get("label", [])))

        placement_counter, placement_count = _analyze_placement(
            xyz,
            seg.astype(np.int32),
            depth_regions,
            placement_geometry,
        )

        if raw_regions == 0:
            status = "no_raw_regions"
        elif shape_count == 0:
            status = "no_shape_regions"
        elif depth_count == 0:
            status = "no_depth_regions"
        elif placement_count == 0:
            status = "no_placement_regions"
        else:
            status = "ok"

        return {
            "image": str(imname),
            "status": status,
            "raw_regions": raw_regions,
            "raw_pixels": raw_pixels,
            "shape_regions": shape_count,
            "depth_regions": depth_count,
            "placement_regions": placement_count,
            "region_events": Counter(e.get("event", "unknown") for e in events),
            "placement_events": placement_counter,
        }
    except Exception as exc:
        return {
            "image": str(imname),
            "status": "exception",
            "raw_regions": 0,
            "raw_pixels": 0,
            "shape_regions": 0,
            "depth_regions": 0,
            "placement_regions": 0,
            "region_events": Counter({"exception": 1}),
            "placement_events": Counter(),
            "error": repr(exc),
        }
    finally:
        TextRegions.stats_collector = old_collector


def _analyze_placement(xyz, seg, regions, placement_geometry=None):
    counter = Counter()
    if regions is None or len(regions.get("label", [])) == 0:
        counter["no_depth_regions"] += 1
        return counter, 0

    labels = np.asarray(regions.get("label", []), dtype=np.int32)
    coeffs = regions.get("coeff", None)
    if coeffs is None:
        counter["missing_coeff"] += len(labels)
        return counter, 0

    coeffs = np.asarray(coeffs)
    kept = 0
    placement_geometry = placement_geometry or {}
    persp_strength = float(placement_geometry.get("persp_strength", 1.0))
    persp_max_tilt_deg = placement_geometry.get("persp_max_tilt_deg", None)

    for i, lbl in enumerate(labels):
        if i >= len(coeffs):
            counter["coeff_index_out_of_range"] += 1
            continue

        mask = (seg == int(lbl)).astype("uint8")
        try:
            res = get_text_placement_mask(
                xyz,
                mask,
                coeffs[i],
                pad=2,
                viz=False,
                persp_strength=persp_strength,
                max_tilt_deg=persp_max_tilt_deg,
            )
        except Exception:
            counter["placement_exception"] += 1
            continue

        if res is None:
            counter["placement_mask_none"] += 1
            continue

        place_mask_fp, _H, _Hinv = res
        if place_mask_fp is None or place_mask_fp.size == 0 or int(place_mask_fp.sum()) < 50:
            counter["placement_too_small_rectified_mask"] += 1
            continue

        counter["placement_kept"] += 1
        kept += 1

    return counter, kept


def _print_summary(totals, file_totals, region_events, placement_events, rows):
    print("\n[RANSAC-STATS] summary")
    print(f"  images: {int(totals.get('images', 0))}")
    for key in ["ok", "no_raw_regions", "no_shape_regions", "no_depth_regions", "no_placement_regions", "exception"]:
        print(f"  {key}: {int(totals.get(key, 0))}")
    for key in ["file_open_failed", "file_parse_failed", "file_no_common_keys"]:
        if totals.get(key, 0):
            print(f"  {key}: {int(totals.get(key, 0))}")

    print("\n[RANSAC-STATS] per-file")
    for path, counters in file_totals.items():
        status = ", ".join(f"{k}={v}" for k, v in counters.items())
        print(f"  {path}: {status}")

    print("\n[RANSAC-STATS] file-level issues")
    for path, counters in file_totals.items():
        if counters.get("file_open_failed", 0):
            print(f"  {path}: file_open_failed")
            continue
        if counters.get("file_parse_failed", 0):
            print(f"  {path}: file_parse_failed")
            continue
        if counters.get("file_no_common_keys", 0):
            print(f"  {path}: file_no_common_keys")
            continue

        images = int(counters.get("images", 0))
        ok = int(counters.get("ok", 0))
        if images > 0 and ok == 0:
            failures = Counter({k: v for k, v in counters.items() if k not in ("images", "ok")})
            dominant = failures.most_common(1)[0][0] if failures else "unknown"
            print(f"  {path}: all sampled images rejected, dominant={dominant}")

    print("\n[RANSAC-STATS] region events")
    for key, val in region_events.most_common():
        print(f"  {key}: {int(val)}")

    print("\n[RANSAC-STATS] placement events")
    for key, val in placement_events.most_common():
        print(f"  {key}: {int(val)}")

    print("\n[RANSAC-STATS] worst images")
    bad_rows = [r for r in rows if r.get("status") != "ok"]
    for row in bad_rows[:20]:
        err = f" error={row['error']}" if row.get("error") else ""
        print(
            f"  {row['status']}: {row['image']} "
            f"raw={row['raw_regions']} shape={row['shape_regions']} "
            f"depth={row['depth_regions']} placement={row['placement_regions']}{err}"
        )
