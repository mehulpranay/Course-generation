"""
Pydantic schema for STEMI v2 (course generation).

v2 is a clean break from v1's per-lesson pipeline, not an extension of it.
Differences that matter for how this file reads:

- No CourseSkeleton / LessonSkeletonItem. v1 split generation into a
  skeleton call (titles, learning targets, introduces/prerequisites tags)
  followed by one call per lesson. v2 generates the whole course in two
  calls total (see generation.py) -- there's no separate skeleton step to
  tag, so `introduces` and `prerequisites` are gone from Lesson entirely.
  Two independent test runs (manual, via a frontier chat model with
  search) produced clean lesson-to-lesson sequencing without this
  scaffolding, so it's dropped rather than carried forward unused.

- Per-claim tagging is KEPT, deliberately, even though no Stage 3 checker
  runs against it yet. Two things still need it: the PDF review renderer
  (json_to_pdf.py colors claims by type so a human reviewer can scan
  straight to what's worth scrutinizing), and whatever Stage 3 checking
  gets built later -- atomic per-claim verification is cheaper and more
  reliable than judging a whole document at once, but only if the claims
  are already separated out at generation time.

- Citation now carries `retrieved_snippet` in addition to source/url/
  detail. v2's draft-generation call has live web search -- when the
  model backs a claim with something it found, capturing what it actually
  found (not just a URL) means a future grounding check can compare the
  claim against the snippet directly, instead of re-fetching the source
  from scratch. Costs nothing to capture now; expensive to reconstruct
  after the fact.

- ToolSpec.capabilities_summary is a STARTING POINT, not verified ground
  truth the way it was in v1. v1 had no search, so a human had to supply
  verified capabilities up front or the model would invent them (the
  CreateAI/audio failure this field exists to prevent). v2's draft call
  can search, so the summary is handed in as a prior for the model to
  check and update, not something to trust blindly. Whatever ends up in
  the final Citation on a hard_fact tool claim is what actually got
  verified.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    MECHANISTIC = "mechanistic"
    HARD_FACT = "hard_fact"
    CAUSAL = "causal"
    PEDAGOGICAL = "pedagogical"


class Citation(BaseModel):
    source: str = Field(..., description="Name of the source, e.g. 'AI4K12', 'CSTA', a specific paper/vendor doc")
    detail: Optional[str] = Field(
        None, description="Specific section/guideline/page this claim traces to, not just the source name"
    )
    url: Optional[str] = Field(None, description="Link to the source, when available")
    retrieved_snippet: Optional[str] = Field(
        None,
        description=(
            "The actual passage the model found via web search that backs this claim, verbatim if "
            "possible -- not the claim restated, the source text itself. Left empty if the claim wasn't "
            "search-verified (e.g. it came from a pre-supplied ToolSpec summary rather than a live search)."
        ),
    )


class ToolSpec(BaseModel):
    """
    A named third-party tool/software/hardware referenced in the course.

    capabilities_summary is supplied by Stage 1 as a starting point. v2's
    draft-generation call has web search and is instructed to verify and,
    if needed, correct or extend this summary -- it is not assumed correct
    on its own the way it had to be in v1.
    """

    name: str
    capabilities_summary: str = Field(
        ...,
        description=(
            "What this tool is believed to do, stated specifically enough to rule things out. "
            "Treated as a prior to verify via search during generation, not as settled fact."
        ),
    )
    citation: Optional[Citation] = Field(None, description="Source backing capabilities_summary, if one exists yet")


class Claim(BaseModel):
    text: str = Field(..., description="The exact claim as it appears in the lesson content")
    claim_type: ClaimType
    citation: Optional[Citation] = Field(
        None, description="Required if claim_type == hard_fact; must be omitted otherwise"
    )


class AssessmentQuestion(BaseModel):
    question: str
    options: List[str] = Field(..., min_length=2)
    correct_answer: str
    supporting_claim_index: Optional[int] = Field(
        None, description="Index into the lesson's claims list that justifies the correct answer, if applicable"
    )


class Lesson(BaseModel):
    lesson_number: int
    title: str
    learning_target: str
    content: str = Field(..., description="Main explanatory content of the lesson")
    activity: str = Field(..., description="Hands-on activity or exercise for this lesson")
    claims: List[Claim] = Field(default_factory=list)
    assessment: Optional[AssessmentQuestion] = None
    equipment_used: List[str] = Field(default_factory=list)


class CourseInput(BaseModel):
    """What Stage 1 hands off to Stage 2 -- the decided candidate plus its grounding."""

    category: str = Field(..., description="e.g. 'AI Literacy'")
    grade_band: str = Field(..., description="e.g. '9-12'")
    course_title: str
    course_outcome: str = Field(
        ...,
        description=(
            "What a student can actually DO at the end of this course, in plain language. "
            "Written or approved by the human in Stage 1."
        ),
    )
    theme: str = Field(..., description="Narrative wrapper, e.g. 'wake-word / physical computing'")
    technique: str = Field(..., description="e.g. 'Physical computing / embedded AI (TinyML)'")
    equipment: List[str] = Field(default_factory=list)
    third_party_software: List[ToolSpec] = Field(default_factory=list)
    grounding_sources: List[Citation] = Field(
        default_factory=list,
        description="Links Stage 1 used to justify this course recommendation -- carried forward for review.",
    )
    pedagogical_grounding: str = Field(
        ..., description="Stage 1's citation-backed reasoning for why this fits this grade band"
    )
    num_lessons: int = 15


class FullCourse(BaseModel):
    course_title: str
    grade_band: str
    course_description: str
    equipment_and_training: str
    lessons: List[Lesson]