import json
from pathlib import Path

from lens_locator.cli import main as cli_main
from lens_locator.demo import create_demo_topography_pair
from lens_locator.result import LensDetection
from lens_locator.topography import RefractiveTopographyEstimator, TopographyConfig


def test_refractive_topography_estimates_synthetic_power(tmp_path: Path):
    config = TopographyConfig(max_pair_distance_px=24.0)
    reference_path, measured_path = create_demo_topography_pair(
        tmp_path,
        sphere_d=0.4,
        cylinder_d=0.2,
        axis_deg=35.0,
        config=config,
    )
    detection = LensDetection(
        bbox_xyxy=(64.0, 44.0, 296.0, 276.0),
        center_xy=(180.0, 160.0),
        radius_px=116.0,
        area_px=42000.0,
        confidence=1.0,
        image_size=(360, 320),
    )

    power_map = RefractiveTopographyEstimator(config).measure(measured_path, reference_path, detection)

    assert len(power_map.samples) >= 20
    assert abs(power_map.sphere_equivalent_d - 0.4) < 0.08
    assert abs(power_map.cylinder_d - 0.2) < 0.08
    assert power_map.rms_fit_error_d < 0.25


def test_cli_outputs_localization_and_topography_json(tmp_path: Path, capsys):
    config = TopographyConfig(max_pair_distance_px=24.0)
    reference_path, measured_path = create_demo_topography_pair(tmp_path, config=config)
    output_json = tmp_path / "result.json"

    exit_code = cli_main(
        [
            str(measured_path),
            "--backend",
            "classical",
            "--measure-topography",
            "--reference",
            str(reference_path),
            "--json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["detections"]
    assert payload["topography"]["metadata"]["matched_spot_count"] >= 10
    capsys.readouterr()
