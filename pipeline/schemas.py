"""Pydantic models for every LLM output. Anything that fails validation is retried once."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    id: str
    claim: str
    quote: str = Field(description="verbatim sentence from the paper that states the claim")
    evidence_hint: str = ""


class ClaimsOut(BaseModel):
    claims: list[Claim] = Field(default_factory=list, max_length=6)


class QuoteNote(BaseModel):
    quote: str
    note: str


class ComplexArg(BaseModel):
    original: str
    plain: str


class Definition(BaseModel):
    term: str
    quote: str
    plain: str


class SectionAnalysis(BaseModel):
    supports_claims: list[str] = Field(default_factory=list)
    difficulty: int = Field(ge=1, le=5)
    difficulty_reason: str = ""
    key_points: list[QuoteNote] = Field(default_factory=list)
    complex_arguments: list[ComplexArg] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    self_check: list[str] = Field(default_factory=list)

    @field_validator("difficulty", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            return max(1, min(5, int(round(float(v)))))
        except (TypeError, ValueError):
            return v


Bucket = Literal["study", "read", "skim", "skip"]


class PlanItem(BaseModel):
    section_id: str
    bucket: Bucket
    why: str = ""


class ReadingPlan(BaseModel):
    summary: str = ""
    order: list[PlanItem]
    prerequisites: list[str] = Field(default_factory=list)
