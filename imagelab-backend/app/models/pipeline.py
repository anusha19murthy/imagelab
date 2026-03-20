from pydantic import BaseModel


class PipelineStep(BaseModel):
    type: str
    params: dict = {}


class PipelineRequest(BaseModel):
    image: str
    image_format: str = "png"
    pipeline: list[PipelineStep]
    include_intermediates: bool = False


class StepTiming(BaseModel):
    step: int
    operator_type: str
    duration_ms: float


class PipelineTimings(BaseModel):
    total_ms: float
    steps: list[StepTiming]


class ChannelHistogram(BaseModel):
    name: str
    mean: float
    std: float
    min: int
    max: int
    histogram: list[int]


class HistogramData(BaseModel):
    channels: list[ChannelHistogram]


class IntermediateStepResult(BaseModel):
    step: int
    operator_type: str
    image: str  # base64-encoded
    histogram: HistogramData


class PipelineResponse(BaseModel):
    success: bool
    image: str | None = None
    image_format: str | None = None
    error: str | None = None
    step: int | None = None
    timings: PipelineTimings | None = None
    intermediates: list[IntermediateStepResult] | None = None
