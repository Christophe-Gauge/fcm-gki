"""Command-line interface: ``fcm-gki image.png -c 5 --spatial --gki``."""

from __future__ import annotations

import argparse
import os

from . import __version__
from .core import ModifiedFCM, render_clusters, render_segmented


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fcm-gki",
        description="Modified Fuzzy C-Means image segmentation "
                    "(spatial centre-of-gravity term + G_ki neighbourhood factor).")
    ap.add_argument("image", help="input image (any format Pillow reads)")
    ap.add_argument("-c", "--clusters", type=int, default=5, help="number of clusters (default 5)")
    ap.add_argument("-m", "--fuzziness", type=float, default=2.0, help="fuzziness m > 1 (default 2)")
    ap.add_argument("--spatial", action="store_true",
                    help="apply spatial distance to cluster centre of gravity")
    ap.add_argument("--gki", action="store_true", help="apply the G_ki neighbourhood factor")
    ap.add_argument("-w", "--window", type=int, default=1,
                    help="G_ki neighbourhood radius, 1 = 3x3 (default 1)")
    ap.add_argument("--max-iter", type=int, default=20, help="maximum iterations (default 20)")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="stop when max membership change < tol (default 1e-5)")
    ap.add_argument("--init", choices=["top_colours", "kmeans++", "random"], default="top_colours",
                    help="centre seeding (default top_colours, as the original)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducible runs")
    ap.add_argument("--grey", action="store_true", help="cluster on greyscale intensity")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="downscale so the longest side is at most this many pixels")
    ap.add_argument("--no-markers", action="store_true",
                    help="don't mark cluster centres when --spatial is on")
    ap.add_argument("-o", "--out-dir", default=".", help="output folder (default: current)")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .io import load_image, save_image  # needs Pillow

    order = "L" if args.grey else "RGB"
    image = load_image(args.image, args.max_dim, order)
    fcm = ModifiedFCM(args.clusters, args.fuzziness, args.spatial, args.gki, args.window,
                      args.max_iter, args.tol, args.init, args.seed, verbose=not args.quiet)
    res = fcm.fit(image)

    os.makedirs(args.out_dir, exist_ok=True)
    mark = args.spatial and not args.no_markers
    save_image(render_segmented(res, mark, marker_colour=(255, 0, 0) if not args.grey else None),
               os.path.join(args.out_dir, "segmented.png"))
    for k, img in enumerate(render_clusters(image, res)):
        save_image(img, os.path.join(args.out_dir, f"Cluster{k}.png"))
    state = "converged" if res.converged else "stopped at max iterations"
    print(f"Done in {res.iterations} iterations ({state}) -> {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
