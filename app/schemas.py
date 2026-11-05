"""Request and response shapes for the API (FR-6, FR-7, FR-8)."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

MAX_ENGINE_ID = 2_147_483_647  # the largest 32-bit integer, so SQLite can never overflow

# A whole number from 1 up to MAX_ENGINE_ID. Strict means "1" and 1.0 are rejected.
EngineId = Annotated[
    int,
    Field(strict=True, ge=1, le=MAX_ENGINE_ID, description="Engine ID, a positive whole number"),
]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_id: EngineId = Field(examples=[1])


class EngineSummary(BaseModel):
    engine_id: int
    n_cycles: int = Field(description="Number of operating cycles recorded for this engine")


class PredictResponse(BaseModel):
    engine_id: int
    cycle: int = Field(description="The engine's latest recorded cycle")
    predicted_rul: float = Field(
        description=(
            "Predicted remaining useful life in cycles. The model was trained on a target "
            "capped at 125, so a value near 125 means 'at least about 125'."
        )
    )


class ErrorResponse(BaseModel):
    detail: str
