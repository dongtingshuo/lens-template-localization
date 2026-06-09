from pathlib import Path

from PIL import Image, ImageDraw

from lens_locator.classical import ClassicalLensDetector


def test_classical_detector_finds_synthetic_lens(tmp_path: Path):
    image_path = tmp_path / "lens.png"
    image = Image.new("RGB", (240, 200), (210, 210, 214))
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 25, 195, 175), fill=(198, 200, 210), outline=(105, 108, 122), width=6)
    image.save(image_path)

    result = ClassicalLensDetector().predict(image_path)

    assert len(result) == 1
    detection = result[0]
    assert 105 <= detection.center_xy[0] <= 135
    assert 85 <= detection.center_xy[1] <= 115
    assert detection.radius_px > 55

