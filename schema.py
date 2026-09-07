"""
Pydantic schema for STEMI Stage 2 (course/lesson generation).

Design notes (from the scoping discussion):
- Stage 1 has already decided the grade-band fit for this course (citation-backed,
  pedagogical-convention-grounded). Stage 2 does NOT re-litigate that.
- Every individual claim inside a lesson is tagged by type, because different
  claim types need different (or no) verification:
    - MECHANISTIC: how something technically works (e.g. "a microphone converts
      sound into a numerical feature vector"). No external citation required,
      but still gets a logical-consistency check downstream (Stage 3).
    - HARD_FACT: a real product name, a real study/number, a real external
      fact. REQUIRES a citation.
    - CAUSAL: an "A because B" claim. No external citation required, but
      REQUIRES a logical-validity check downstream (CoVe-style).
    - PEDAGOGICAL: framing/transition/narrative text with no checkable claim
      in it at all. No check needed.
- Lesson-to-lesson sequencing (does lesson 6 build correctly on lesson 5) is a
  separate, lightweight internal-consistency concern — not a citation concern.
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


class ToolSpec(BaseModel):
    """
    A named third-party tool/software/hardware referenced in the course.

    This exists because of a real failure: a prior generation confidently
    wrote 4+ lessons of instructions for "micro:bit CreateAI" to record audio
    and train a wake-word model — CreateAI is accelerometer/movement-only and
    has no audio capability at all. The model had only a tool NAME to work
    from and filled in plausible-sounding capabilities from memory.

    capabilities_summary is the ONLY source of truth Stage 2 is allowed to
    assume about what this tool can do. It must be populated (ideally by
    Stage 1's research, or verified separately) BEFORE Stage 2 runs — never
    left for the lesson-generation model to infer.
    """

    name: str
    capabilities_summary: str = Field(
        ...,
        description=(
            "What this tool can ACTUALLY do, stated specifically enough to rule things out "
            "(e.g. 'Trains a model on accelerometer/motion data only; no audio, microphone, "
            "camera, or image input of any kind.'). Generation must not assume any capability "
            "not stated here."
        ),
    )
    citation: Optional[Citation] = Field(
        None, description="Source backing capabilities_summary — required before this tool is trusted in production, not just for this test."
    )


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


class LessonSkeletonItem(BaseModel):
    lesson_number: int
    title: str
    learning_target: str
    introduces: List[str] = Field(
        default_factory=list,
        description="Concepts/skills/setup-steps this lesson teaches for the FIRST time (e.g. 'connecting micro:bit via USB', 'what an accelerometer is'). Used to check that later lessons don't assume something before it's taught.",
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Concepts/skills/setup-steps this lesson ASSUMES the student already has — each one must match something an EARLIER lesson's `introduces` states, or the sequencing check will reject the skeleton.",
    )


class CourseSkeleton(BaseModel):
    course_title: str
    grade_band: str
    course_description: str
    lessons: List[LessonSkeletonItem]


class Lesson(BaseModel):
    lesson_number: int
    title: str
    learning_target: str
    content: str = Field(..., description="Main explanatory content of the lesson")
    activity: str = Field(..., description="Hands-on activity or exercise for this lesson")
    claims: List[Claim] = Field(default_factory=list)
    assessment: Optional[AssessmentQuestion] = None
    equipment_used: List[str] = Field(default_factory=list)
    introduces: List[str] = Field(
        default_factory=list,
        description="Carried over from this lesson's skeleton entry, so the sequencing tags are visible in the final output/PDF for review.",
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Carried over from this lesson's skeleton entry, so the sequencing tags are visible in the final output/PDF for review.",
    )


class CourseInput(BaseModel):
    """What Stage 1 hands off to Stage 2 — the decided candidate plus its grounding."""

    category: str = Field(..., description="e.g. 'AI Literacy'")
    grade_band: str = Field(..., description="e.g. '9-12'")
    course_title: str
    course_outcome: str = Field(
        ...,
        description=(
            "What a student can actually DO at the end of this course, in plain language — the "
            "destination every lesson builds toward. Written or approved by the human in Stage 1. "
            "Without this, the skeleton has no stated end-state and the model infers one from the "
            "title alone."
        ),
    )
    theme: str = Field(..., description="Narrative wrapper, e.g. 'wake-word / physical computing'")
    technique: str = Field(..., description="e.g. 'Physical computing / embedded AI (TinyML)'")
    equipment: List[str] = Field(default_factory=list)
    third_party_software: List[ToolSpec] = Field(
        default_factory=list,
        description="Structured, capability-verified tools only — never a bare name string.",
    )
    grounding_sources: List[Citation] = Field(
        default_factory=list,
        description="Links Stage 1 used to justify this course recommendation — carried forward so a human can verify Stage 1's reasoning, and as candidate sources for hard_fact grounding during generation.",
    )
    pedagogical_grounding: str = Field(
        ..., description="Stage 1's citation-backed reasoning for why this fits this grade band — carried forward, not re-derived"
    )
    num_lessons: int = 15


class FullCourse(BaseModel):
    course_title: str
    grade_band: str
    course_description: str
    equipment_and_training: str
    lessons: List[Lesson]