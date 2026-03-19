import numpy as np

from app.operators.transformation.laplacian import Laplacian


def test_laplacian_no_uint8_wrapping():
    """Regression test: strong edges must be clipped to 255, not wrapped modulo 256."""
    img = np.zeros((20, 20), dtype=np.uint8)
    img[10:, :] = 255  # sharp horizontal edge
    operator = Laplacian({})
    result = operator.compute(img)
    assert result.dtype == np.uint8, "Output must be uint8"
    assert result.max() == 255, "Expected saturated edge response of 255"
    assert result[5, 0] == 255 or result[10, 0] == 255, "Edge pixel should be saturated to 255"
