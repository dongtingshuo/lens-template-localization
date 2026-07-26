from lens_locator.geometry import (
    detection_from_polygon,
    polygon_area,
    polygon_centroid,
    yolo_segment_to_points,
)


def test_polygon_area_and_centroid_for_square():
    points = [(0, 0), (10, 0), (10, 10), (0, 10)]

    assert polygon_area(points) == 100
    assert polygon_centroid(points) == (5.0, 5.0)


def test_yolo_segment_to_detection():
    values = [0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75]
    points = yolo_segment_to_points(values, width=200, height=100)
    detection = detection_from_polygon(points, image_size=(200, 100))

    assert detection.bbox_xyxy == (50.0, 25.0, 150.0, 75.0)
    assert detection.center_xy == (100.0, 50.0)
    assert detection.radius_px == 37.5

