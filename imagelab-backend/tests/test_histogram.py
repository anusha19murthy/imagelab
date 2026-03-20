import numpy as np

from app.utils.histogram import compute_histogram


def test_grayscale_histogram_has_one_channel(grayscale_image):
    result = compute_histogram(grayscale_image)
    assert len(result["channels"]) == 1
    assert result["channels"][0]["name"] == "Gray"


def test_color_histogram_has_three_channels(color_image):
    result = compute_histogram(color_image)
    assert len(result["channels"]) == 3
    assert [ch["name"] for ch in result["channels"]] == ["Blue", "Green", "Red"]


def test_histogram_bin_count_is_256(color_image):
    result = compute_histogram(color_image)
    for ch in result["channels"]:
        assert len(ch["histogram"]) == 256


def test_statistics_match_numpy(color_image):
    result = compute_histogram(color_image)
    for i, ch in enumerate(result["channels"]):
        channel_data = color_image[:, :, i]
        assert abs(ch["mean"] - float(np.mean(channel_data))) < 1e-6
        assert abs(ch["std"] - float(np.std(channel_data))) < 1e-6
        assert ch["min"] == int(np.min(channel_data))
        assert ch["max"] == int(np.max(channel_data))


def test_uniform_white_image():
    white = np.full((50, 50), 255, dtype=np.uint8)
    result = compute_histogram(white)
    ch = result["channels"][0]
    assert ch["mean"] == 255.0
    assert ch["std"] == 0.0
    assert ch["min"] == 255
    assert ch["max"] == 255
    # All pixels at bin 255
    assert ch["histogram"][255] == 50 * 50
    assert sum(ch["histogram"][:255]) == 0


def test_uniform_black_image():
    black = np.zeros((50, 50), dtype=np.uint8)
    result = compute_histogram(black)
    ch = result["channels"][0]
    assert ch["mean"] == 0.0
    assert ch["min"] == 0
    assert ch["max"] == 0
    assert ch["histogram"][0] == 50 * 50
    assert sum(ch["histogram"][1:]) == 0


def test_histogram_sum_equals_pixel_count(color_image):
    result = compute_histogram(color_image)
    pixel_count = color_image.shape[0] * color_image.shape[1]
    for ch in result["channels"]:
        assert sum(ch["histogram"]) == pixel_count
