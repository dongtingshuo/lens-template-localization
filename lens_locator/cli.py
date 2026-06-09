"""Command line interface for lens localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .config import load_config
from .pipeline import LensLocator, LensLocatorConfig
from .topography import RefractiveTopographyEstimator
from .visualize import save_overlay, save_topography_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lens recognition and localization pipeline.")
    parser.add_argument("source", help="Image file or directory.")
    parser.add_argument("--backend", choices=["auto", "yolo", "classical"], default="auto")
    parser.add_argument("--config", default=None, help="YAML config path.")
    parser.add_argument("--json", dest="json_path", default=None, help="Path for JSON result.")
    parser.add_argument("--overlay-dir", default=None, help="Directory for annotated overlay images.")
    parser.add_argument("--measure-topography", action="store_true", help="Estimate refractive power topography.")
    parser.add_argument("--reference", default=None, help="Reference Hartmann/grid image for topography.")
    parser.add_argument("--topography-overlay-dir", default=None, help="Directory for power-map overlay images.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else LensLocatorConfig()
    config.backend = args.backend
    locator = LensLocator(config)
    images = list(_iter_images(Path(args.source)))
    if not images:
        raise SystemExit(f"No images found: {args.source}")

    results = []
    for image_path in images:
        result = locator.locate(image_path)
        payload = result.to_dict()
        if args.overlay_dir:
            out = Path(args.overlay_dir) / f"{image_path.stem}_overlay.jpg"
            save_overlay(image_path, result, out)
        if args.measure_topography:
            topography = RefractiveTopographyEstimator(config.topography).measure(
                image_path,
                reference_path=args.reference,
                lens_detection=result.best,
            )
            payload["topography"] = topography.to_dict()
            if args.topography_overlay_dir:
                out = Path(args.topography_overlay_dir) / f"{image_path.stem}_topography.jpg"
                save_topography_overlay(image_path, topography, out)
        results.append(payload)

    payload = results[0] if len(results) == 1 else {"results": results}
    text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _iter_images(source: Path) -> Iterable[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if source.is_file() and source.suffix.lower() in suffixes:
        yield source
    elif source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() in suffixes:
                yield path


if __name__ == "__main__":
    raise SystemExit(main())
