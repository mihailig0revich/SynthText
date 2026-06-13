import argparse

from .config import GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Synthetic Scene-Text Images")
    parser.add_argument("--viz", action="store_true", default=False, help="turn on visualizations")
    parser.add_argument("--interactive", action="store_true", default=False, help="ask for input directory at startup")
    parser.add_argument("--input-dir", default=GenerationConfig.input_dir, help="directory with input .h5 files")
    parser.add_argument("--fallback-h5", default=GenerationConfig.fallback_h5, help="fallback input .h5 path")
    parser.add_argument("--render-data-path", default=GenerationConfig.render_data_path, help="text rendering data directory")
    parser.add_argument("--output-file", default=GenerationConfig.output_file, help="base output H5 path")
    parser.add_argument("--png-dir", default=GenerationConfig.png_dir, help="PNG output directory")
    parser.add_argument("--num-img", type=int, default=GenerationConfig.num_img, help="-1 means all images")
    parser.add_argument("--instances-per-image", type=int, default=GenerationConfig.instances_per_image)
    parser.add_argument("--secs-per-img", type=int, default=GenerationConfig.secs_per_img)
    parser.add_argument("--max-global-tries", type=int, default=GenerationConfig.max_global_tries)
    parser.add_argument("--max-h5-size-gb", type=float, default=GenerationConfig.max_h5_size_gb)
    parser.add_argument("--region-workers", type=int, default=GenerationConfig.region_workers,
                        help="threads for independent region plane fitting; 1 keeps serial behavior")
    parser.add_argument("--ransac-debug", action="store_true", default=GenerationConfig.ransac_debug,
                        help="print detailed RANSAC plane-fitting diagnostics")
    parser.add_argument("--ransac-stats", type=int, default=GenerationConfig.ransac_stats,
                        help="analyze RANSAC/region rejection statistics for first N images and exit")
    parser.add_argument("--placement-debug", action="store_true", default=GenerationConfig.placement_debug,
                        help="print text placement and overlay rejection diagnostics")
    parser.add_argument("--debug-progress", action="store_true", default=GenerationConfig.debug_progress,
                        help="print per-file/per-image progress in debug and statistics modes")
    return parser


def config_from_args(args) -> GenerationConfig:
    return GenerationConfig(
        input_dir=args.input_dir,
        fallback_h5=args.fallback_h5,
        render_data_path=args.render_data_path,
        output_file=args.output_file,
        png_dir=args.png_dir,
        num_img=args.num_img,
        instances_per_image=args.instances_per_image,
        secs_per_img=args.secs_per_img,
        max_global_tries=args.max_global_tries,
        max_h5_size_gb=args.max_h5_size_gb,
        region_workers=max(1, args.region_workers),
        ransac_debug=args.ransac_debug,
        ransac_stats=max(0, args.ransac_stats),
        placement_debug=args.placement_debug,
        debug_progress=(
            args.debug_progress
            or args.ransac_debug
            or args.placement_debug
            or args.ransac_stats > 0
        ),
        viz=args.viz,
        interactive=args.interactive,
    )


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    from .pipeline import generate_dataset

    generate_dataset(config_from_args(args))
