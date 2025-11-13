"""Command-line interface for the background remover."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .remover import BackgroundRemover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the background from an image while preserving quality.",
    )
    parser.add_argument("input", type=Path, help="Path to the input image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination for the background-free PNG (defaults to input name with _transparent).",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help=(
            "Optional ONNX Runtime execution providers, e.g. 'CUDAExecutionProvider'"
            " for GPU acceleration."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file {args.input} does not exist")

    image = Image.open(args.input)
    remover = BackgroundRemover(providers=args.providers)
    output = remover.remove_background(image)

    if args.output is None:
        output_path = args.input.with_name(f"{args.input.stem}_transparent.png")
    else:
        output_path = args.output

    output.save(output_path, format="PNG")
    print(f"Saved background-free image to {output_path}")


if __name__ == "__main__":
    main()
