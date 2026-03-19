import numpy as np

from app.operators.transformation.laplacian import Laplacian


def test_laplacian_no_uint8_overflow():
    """Regression test for uint8 overflow fix: Laplacian output must clip values to [0, 255], not wrap modulo 256."""
    # A hard vertical edge produces large second-derivative values
    img = np.zeros((10, 10), dtype=np.uint8)
    img[5, 5] = 255
    op = Laplacian({"ksize": 1})
    result = op.compute(img)
    assert result.dtype == np.uint8
    assert result.max() <= 255
    assert result.max() == 255, "Expected edge pixels to saturate at 255, not wrap"
