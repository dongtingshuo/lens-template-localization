#!/usr/bin/env python3
"""Generate synthetic demo images for the lens and topography pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lens_locator.demo import create_demo_lens_image, create_demo_topography_pair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate demo inputs for lens localization and refractive topography.")
    parser.add_argument("--output-dir", default="outputs/demo", help="Directory for generated images.")
    parser.add_argument("--sphere", type=float, default=0.4, help="Synthetic sphere equivalent in diopters.")
    parser.add_argument("--cylinder", type=float, default=0.2, help="Synthetic cylinder in diopters.")
    parser.add_argument("--axis", type=float, default=35.0, help="Synthetic cylinder axis in degrees.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    lens_path = create_demo_lens_image(out_dir / "lens.png")
    reference_path, measured_path = create_demo_topography_pair(
        out_dir,
        sphere_d=args.sphere,
        cylinder_d=args.cylinder,
        axis_deg=args.axis,
    )
    print(f"lens={lens_path}")
    print(f"reference={reference_path}")
    print(f"measured={measured_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
