# PoC: Per-Step Intermediate Previews with Histogram Analysis

## What this demonstrates

This proof of concept implements the **core mechanism** of the GSoC project: capturing the image state and computing per-channel histogram statistics **after every operator** in an ImageLab pipeline.

Currently, ImageLab's pipeline executor processes a chain of OpenCV operators and returns only the **final** image. Students see no intermediate states, making the tool a black box. This PoC modifies the executor loop so that — when opt-in via `include_intermediates: true` — each completed step also produces:

- A base64-encoded snapshot of the image at that point
- Per-channel (B/G/R or Gray) histogram data (256 bins) with statistics (mean, std, min, max)

This is the foundation that all other proposed features (histogram UI, pipeline persistence, batch processing, macros) build on.

## What was changed

**New file — `app/utils/histogram.py`**
Computes per-channel histograms and pixel statistics for any `np.ndarray` image using `cv2.calcHist` and numpy. Handles both grayscale (1 channel) and colour (3 channels, BGR).

**Modified — `app/models/pipeline.py`**
Added Pydantic models: `ChannelHistogram`, `HistogramData`, `IntermediateStepResult`. Extended `PipelineRequest` with `include_intermediates: bool = False` (opt-in, backward compatible). Extended `PipelineResponse` with `intermediates: list[IntermediateStepResult] | None = None`.

**Modified — `app/services/pipeline_executor.py`**
After each `operator.compute(image)` call, if intermediates are requested, captures the image state and computes its histogram. Partial intermediates are still returned on failure (same pattern as existing partial timings).

**New tests — `tests/test_histogram.py`** (7 tests)
Unit tests for `compute_histogram`: channel count, bin count, statistics accuracy vs numpy, uniform images, pixel count invariant.

**New tests — `tests/test_intermediate_previews.py`** (11 tests)
Integration tests for the intermediate capture mechanism: backward compatibility (None when not requested), correct count (noops excluded), valid base64, histogram structure, 1-indexed steps, pixel-exact match against manual OpenCV, partial results on failure.

## How to run

From the `imagelab-backend/` directory:

```bash
# Install dependencies (if not already done)
uv sync

# Run the PoC tests
uv run pytest tests/test_histogram.py tests/test_intermediate_previews.py -v

# Run ALL tests (to verify nothing is broken)
uv run pytest tests/ -v

# Lint check
uv run ruff check app/utils/histogram.py app/models/pipeline.py app/services/pipeline_executor.py
uv run ruff format --check app/utils/histogram.py app/models/pipeline.py app/services/pipeline_executor.py
```

All 18 new tests pass. All previously-passing tests remain green (the 2 pre-existing failures in `test_pipeline_executor.py` are a known step-indexing mismatch in the upstream test expectations, not caused by this change).

## Design decisions

1. **Opt-in via request flag** — `include_intermediates` defaults to `False`, so the existing API contract is unchanged. No client changes are required to keep current behaviour.

2. **Follows existing patterns** — The histogram utility lives in `app/utils/` alongside `image.py` and `color.py`. The Pydantic models extend the existing `app/models/pipeline.py`. Tests use the same fixtures from `conftest.py`.

3. **No new dependencies** — Uses `cv2.calcHist` and `numpy` which are already in `pyproject.toml`.

4. **Partial results on error** — Just like the existing `timings` field returns partial step timings when a later step fails, `intermediates` returns results for all completed steps before the failure.
