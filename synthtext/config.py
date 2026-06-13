from dataclasses import dataclass
import os.path as osp


@dataclass
class GenerationConfig:
    """Runtime settings for dataset generation."""

    input_dir: str = "input"
    fallback_h5: str = osp.join("street", "bg_data", "bg_data.h5")
    render_data_path: str = "data"
    output_file: str = osp.join("results", "SynthText.h5")
    png_dir: str = "results_png"

    num_img: int = -1
    instances_per_image: int = 1
    secs_per_img: int = 5
    max_global_tries: int = 8
    max_h5_size_gb: float = 10.0
    region_workers: int = 1
    ransac_debug: bool = False
    ransac_stats: int = 0
    placement_debug: bool = False
    debug_progress: bool = False

    viz: bool = False
    interactive: bool = False
