import numpy as np

from app.models.pipeline import PipelineRequest, PipelineStep
from app.services.pipeline_executor import execute_pipeline
from app.utils.image import decode_base64_image


def test_intermediates_not_included_by_default(make_request):
    """Backward compat: intermediates is None when flag is omitted."""
    steps = [PipelineStep(type="imageconvertions_grayimage")]
    res = execute_pipeline(make_request(steps))
    assert res.success is True
    assert res.intermediates is None


def test_intermediates_returned_when_requested(sample_image_b64):
    steps = [PipelineStep(type="imageconvertions_grayimage")]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)
    assert res.success is True
    assert res.intermediates is not None
    assert len(res.intermediates) == 1


def test_intermediate_count_matches_non_noop_steps(sample_image_b64):
    steps = [
        PipelineStep(type="basic_readimage"),  # noop — skipped
        PipelineStep(type="imageconvertions_grayimage"),
        PipelineStep(type="blurring_applygaussianblur", params={"widthSize": 3, "heightSize": 3}),
    ]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)
    assert res.success is True
    # Only the 2 non-noop steps produce intermediates
    assert len(res.intermediates) == 2


def test_intermediate_images_are_valid_base64(sample_image_b64):
    steps = [
        PipelineStep(type="imageconvertions_grayimage"),
        PipelineStep(type="blurring_applygaussianblur", params={"widthSize": 3, "heightSize": 3}),
    ]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)
    for inter in res.intermediates:
        img = decode_base64_image(inter.image)
        assert img is not None
        assert img.size > 0


def test_intermediate_has_histogram_data(sample_image_b64):
    steps = [PipelineStep(type="imageconvertions_grayimage")]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)
    inter = res.intermediates[0]
    # Grayscale output should have 1 histogram channel
    assert len(inter.histogram.channels) == 1
    assert inter.histogram.channels[0].name == "Gray"
    assert len(inter.histogram.channels[0].histogram) == 256


def test_color_intermediate_has_three_histogram_channels(sample_image_b64):
    # Blur preserves colour channels
    steps = [PipelineStep(type="blurring_applygaussianblur", params={"widthSize": 3, "heightSize": 3})]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)
    inter = res.intermediates[0]
    assert len(inter.histogram.channels) == 3
    assert [ch.name for ch in inter.histogram.channels] == ["Blue", "Green", "Red"]


def test_intermediate_step_numbers_are_1_indexed(sample_image_b64):
    steps = [
        PipelineStep(type="imageconvertions_grayimage"),
        PipelineStep(type="blurring_applygaussianblur", params={"widthSize": 3, "heightSize": 3}),
    ]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)
    assert res.intermediates[0].step == 1
    assert res.intermediates[1].step == 2


def test_intermediate_operator_types_are_correct(sample_image_b64):
    steps = [
        PipelineStep(type="imageconvertions_grayimage"),
        PipelineStep(type="blurring_applygaussianblur", params={"widthSize": 3, "heightSize": 3}),
    ]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)
    assert res.intermediates[0].operator_type == "imageconvertions_grayimage"
    assert res.intermediates[1].operator_type == "blurring_applygaussianblur"


def test_intermediate_image_matches_manual_opencv(sample_image_b64):
    """The intermediate after grayscale should pixel-match manual cv2 conversion."""
    import cv2

    steps = [PipelineStep(type="imageconvertions_grayimage")]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)

    # Decode the intermediate image
    actual = decode_base64_image(res.intermediates[0].image)

    # Manually apply the same operator
    original = decode_base64_image(sample_image_b64)
    expected = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    assert np.array_equal(actual, expected)


def test_partial_intermediates_on_failure(sample_image_b64):
    """Intermediates for completed steps are still returned when a later step fails."""
    steps = [
        PipelineStep(type="imageconvertions_grayimage"),
        PipelineStep(type="not_a_real_operator"),
    ]
    req = PipelineRequest(image=sample_image_b64, pipeline=steps, include_intermediates=True)
    res = execute_pipeline(req)
    assert res.success is False
    # First step succeeded, so we get its intermediate
    assert len(res.intermediates) == 1
    assert res.intermediates[0].operator_type == "imageconvertions_grayimage"


def test_empty_pipeline_returns_empty_intermediates(sample_image_b64):
    req = PipelineRequest(image=sample_image_b64, pipeline=[], include_intermediates=True)
    res = execute_pipeline(req)
    assert res.success is True
    assert res.intermediates == []
