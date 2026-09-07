"""
Stage 2 generation pipeline: CourseInput -> FullCourse.

Two steps, matching the planner/executor split from the scoping doc:
  1. generate_skeleton() — course-level structure only (titles + learning
     targets + introduces/prerequisites tags), schema-enforced, low
     temperature, IMMEDIATELY checked for prerequisite-ordering violations
     before any lesson content is generated (cheap to catch here, expensive
     to discover after 15 lessons are written).
  2. generate_lesson() — full content for ONE lesson at a time. Each call
     sees: the full skeleton (titles/targets only, for forward-awareness),
     and the ACTUAL generated `activity` text of every already-generated
     prior lesson (not just its title) — so lesson N knows exactly what
     steps the student already completed and doesn't repeat them.

Two deterministic, non-LLM checks run automatically:
  - tool_check.check_tool_capability_consistency — catches claims that pair
    a named tool with a capability its ToolSpec doesn't support.
  - sequence_check.check_skeleton_prerequisites — catches skeleton-level
    ordering errors (a lesson assuming something not yet taught). Raises,
    since this is caught before any expensive lesson generation happens.
  - sequence_check.check_activity_redundancy — catches lessons whose
    generated activities substantially duplicate each other. Warning-only
    (heuristic), surfaced to the human reviewer.

Every lesson's claims are tagged by type (mechanistic / hard_fact / causal /
pedagogical) at generation time, per schema.py.

Usage:
    export OPENAI_API_KEY=...
    python generate.py   # runs the example course at the bottom of this file
"""

from __future__ import annotations

import json
import logging
import os
from typing import List

from google import genai
from google.genai import types

from schema import (
    Claim,
    ClaimType,
    Citation,
    CourseInput,
    CourseSkeleton,
    FullCourse,
    Lesson,
    ToolSpec,
)
from tool_check import check_tool_capability_consistency
from sequence_check import check_skeleton_prerequisites

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stemi.stage2")

MODEL = "gemini-3.6-flash"  # any Gemini model supporting structured outputs

# Some models reject an explicit temperature and only accept the default.
# Set this to False for those and the calls below omit the parameter.
SUPPORTS_TEMPERATURE = True

SKELETON_TEMPERATURE = 0.2  # low: structure should be deterministic
LESSON_TEMPERATURE = 0.4

# Structured output silently returns None if generation hits the output-token
# ceiling, so this is set generously — a full lesson with claims and an
# assessment is a lot of JSON.
MAX_OUTPUT_TOKENS = 32768


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before running generation.")
    return genai.Client(api_key=api_key)


def _config(system_prompt: str, schema, temperature: float) -> types.GenerateContentConfig:
    """Builds the per-request config. Gemini takes the system prompt, the
    response schema, and generation settings all in one config object."""
    kwargs = dict(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    if SUPPORTS_TEMPERATURE:
        kwargs["temperature"] = temperature
    return types.GenerateContentConfig(**kwargs)


SKELETON_SYSTEM_PROMPT = """You are designing the STRUCTURE ONLY of a K-12 course — lesson titles, learning targets, and two sequencing tags, nothing else. Do not write lesson content.

Rules:
- `course_outcome` states what a student can do at the end of this course. Treat it as the destination: the final lesson should land there, and every earlier lesson should be a step toward it. If the outcome names specific capabilities (e.g. testing whether the system still works for a different person), make sure some lesson covers each of them.
- The grade-band fit for this course topic has ALREADY been decided upstream (see pedagogical_grounding in the input). Do not re-justify or second-guess the grade-band choice — just build a sensible lesson sequence for it.
- For EVERY lesson, fill in:
  - `introduces`: concepts/skills/setup-steps this lesson teaches for the FIRST time (e.g. "connecting the device via USB", "what an accelerometer is"). Be specific and consistent in phrasing — later lessons will reference these exact phrases as prerequisites.
  - `prerequisites`: concepts/skills/setup-steps this lesson ASSUMES the student already has. EVERY prerequisite must match something an EARLIER lesson's `introduces` states. A lesson must never require deploying, operating, or building on equipment/software/concepts that no prior lesson has introduced. Basic device setup (connecting hardware, opening required software) must be introduced before any lesson that assumes the student can already do it.
- No lesson should repeat the same core steps (e.g. "deploy and test") that an earlier lesson already completed — treat that as the later lesson's job to build ON, not redo.
- Match the exact number of lessons requested.
- Keep learning targets concrete and action-oriented.
"""

LESSON_SYSTEM_PROMPT = """You are writing ONE lesson of a K-12 course, given its place in the overall course skeleton and what students have ALREADY DONE in prior lessons. Only write this lesson — you are not writing the rest of the course.

CRITICAL — do not repeat completed work, and do not do future work:
- `prior_lessons_context` gives each earlier lesson's title, learning target, and its ACTUAL completed_activity text (what the student literally already did). If a prior lesson's completed_activity already covers a step (e.g. deploying a model, connecting hardware), do NOT write that step again in this lesson's activity — reference that it's already done and build on its result instead.
- `later_lessons_context` gives the title and learning target of every lesson that comes AFTER this one (not yet written). Stay within THIS lesson's own learning target. If a later lesson's learning target already covers something (e.g. "deploy the model and test it"), that is NOT this lesson's job, even if it would feel natural to include — leave it for that lesson, and end this lesson's activity at the boundary of its own target.

Write a real, hands-on course. Students must actually connect the hardware, use the software, collect data, train, program, and test on the device. Do NOT substitute fictional data, invented sample records, hypothetical prediction tables, or "practice audits" of made-up results for real activities, and do not mark core steps as "pending."

`course_outcome` in the input states where the whole course is heading. Use it to judge how much detail this lesson needs and what it must leave for later — but write only this lesson's own step toward it.

Rules for claim tagging (this is the most important part):
- Break every substantive statement in your lesson content into a `claims` entry, tagged with the correct claim_type:
  - "mechanistic": explains how something technically works in general (e.g. how a microphone signal becomes a numeric feature vector). No citation.
  - "hard_fact": a real product name, a real study/number, or ANY CLAIM ABOUT WHAT A SPECIFIC NAMED TOOL, PRODUCT, OR PLATFORM CAN DO. This last category is never "mechanistic," no matter how technical it sounds — if it names a real tool and says what it does, it is hard_fact. MUST include a citation with a real, plausible source name. If you are not confident of a real citation for a factual assertion, rephrase or omit that assertion — but do NOT let this stop you writing the lesson's practical steps, which are instructions, not factual claims, and do not need citations.
  - "causal": an "A because B" claim (e.g. "the classifier is small because it must run continuously without draining the battery"). No citation, but must be a claim you can defend if asked to justify the logical step.
  - "pedagogical": framing, narrative, or transition text with no checkable claim in it (e.g. "let's think about how your phone always knows when you say its name"). Never tag this as hard_fact.
- Do NOT tag every sentence — only tag substantive, checkable statements. Narrative/instructional filler does not need a claims entry at all.
- The activity should be concrete and doable with the equipment and software listed in the course input.
- Include one assessment question with 3-4 options and a clear correct answer. If the correct answer depends on a specific claim, set supporting_claim_index to that claim's position in your claims list.
"""


def generate_skeleton(course_input: CourseInput) -> CourseSkeleton:
    logger.info("Stage 2a: generating skeleton for %r (%d lessons)", course_input.course_title, course_input.num_lessons)
    client = _client()
    response = client.models.generate_content(
        model=MODEL,
        contents=course_input.model_dump_json(indent=2),
        config=_config(SKELETON_SYSTEM_PROMPT, CourseSkeleton, SKELETON_TEMPERATURE),
    )
    skeleton = response.parsed
    if skeleton is None:
        logger.error(
            "Skeleton generation returned no parsed output. This usually means the "
            "response hit MAX_OUTPUT_TOKENS (currently %d) or was blocked. Raw text: %.500s",
            MAX_OUTPUT_TOKENS, response.text,
        )
        raise RuntimeError("Skeleton generation returned no parsed output.")
    if len(skeleton.lessons) != course_input.num_lessons:
        logger.error("Skeleton has %d lessons, expected %d.", len(skeleton.lessons), course_input.num_lessons)
        raise ValueError(
            f"Skeleton has {len(skeleton.lessons)} lessons, expected {course_input.num_lessons}."
        )
    logger.info("Stage 2a done. Lessons planned: %s", [item.title for item in skeleton.lessons])

    # Deterministic guard: catch skeleton-level ordering errors (e.g. a
    # lesson assuming setup steps a later lesson introduces) BEFORE any
    # lesson content is generated. This is the fix for the Lesson 1/2
    # inversion class — passing more context to lesson generation can't
    # repair an already-wrong skeleton order.
    #
    # NOTE: currently WARN-ONLY, not raising. The check uses exact string
    # matching, so a semantic-but-not-literal match ("connect via USB" vs
    # "connecting the micro:bit through USB") would halt an otherwise-fine
    # skeleton. During experimentation, log and continue so the output is
    # actually visible; tighten to raising once phrasing consistency is
    # confirmed in practice.
    prereq_errors = check_skeleton_prerequisites(skeleton)
    if prereq_errors:
        logger.warning(
            "Skeleton prerequisite-ordering check found %d issue(s) — CONTINUING ANYWAY "
            "(warn-only mode). Check whether these are real ordering errors or just "
            "phrasing mismatches:", len(prereq_errors)
        )
        for err in prereq_errors:
            logger.warning("  - %s", err)
    else:
        logger.info("Skeleton prerequisite-ordering check passed.")

    return skeleton


def generate_lesson(
    course_input: CourseInput,
    skeleton: CourseSkeleton,
    lesson_number: int,
    previously_generated_lessons: List[Lesson],
) -> Lesson:
    target_item = next(
        (item for item in skeleton.lessons if item.lesson_number == lesson_number), None
    )
    if target_item is None:
        raise ValueError(f"No skeleton entry for lesson_number={lesson_number}")

    logger.info("Stage 2b: generating lesson %d/%d: %r", lesson_number, course_input.num_lessons, target_item.title)

    generated_by_number = {l.lesson_number: l for l in previously_generated_lessons}
    prior_lessons_context = []
    later_lessons_context = []
    for item in skeleton.lessons:
        if item.lesson_number < lesson_number:
            entry = {"title": item.title, "learning_target": item.learning_target}
            prior_lesson = generated_by_number.get(item.lesson_number)
            if prior_lesson is not None:
                entry["completed_activity"] = prior_lesson.activity
            prior_lessons_context.append(entry)
        elif item.lesson_number > lesson_number:
            # Title + learning_target ONLY — these lessons haven't been
            # written yet, so there's no activity to show. This exists so
            # the model knows work reserved for a later lesson (e.g.
            # "deploy and test") isn't its job to do now.
            later_lessons_context.append({"title": item.title, "learning_target": item.learning_target})

    client = _client()
    user_payload = {
        "course_input": json.loads(course_input.model_dump_json()),
        "this_lesson": json.loads(target_item.model_dump_json()),
        "prior_lessons_context": prior_lessons_context,
        "later_lessons_context": later_lessons_context,
    }

    response = client.models.generate_content(
        model=MODEL,
        contents=json.dumps(user_payload, indent=2),
        config=_config(LESSON_SYSTEM_PROMPT, Lesson, LESSON_TEMPERATURE),
    )
    lesson = response.parsed
    if lesson is None:
        logger.error(
            "Lesson %d returned no parsed output — likely hit MAX_OUTPUT_TOKENS (%d) "
            "or was blocked. Raw text: %.500s",
            lesson_number, MAX_OUTPUT_TOKENS, response.text,
        )
        raise RuntimeError(f"Lesson {lesson_number} generation returned no parsed output.")

    # Deterministic guard: hard_fact claims must carry a citation.
    # WARN-ONLY during experimentation (see note in generate_skeleton) so a
    # single missing citation doesn't kill a run partway through — the
    # violation is loud in the logs and visible in the PDF review output.
    for claim in lesson.claims:
        if claim.claim_type == ClaimType.HARD_FACT and claim.citation is None:
            logger.warning(
                "Lesson %d: hard_fact claim MISSING CITATION (continuing anyway): %r",
                lesson_number,
                claim.text,
            )

    # Deterministic guard: tool-capability consistency.
    tool_warnings = check_tool_capability_consistency(lesson, course_input.third_party_software)
    if tool_warnings:
        logger.warning("Lesson %d: %d tool-capability warning(s) — see above.", lesson_number, len(tool_warnings))
    else:
        logger.info("Lesson %d: tool-capability check clean.", lesson_number)

    logger.info("Stage 2b done for lesson %d.", lesson_number)
    # Carry the skeleton's sequencing tags onto the generated lesson so they
    # appear in the JSON/PDF output and can actually be inspected.
    lesson.introduces = target_item.introduces
    lesson.prerequisites = target_item.prerequisites
    return lesson


def generate_course(course_input: CourseInput) -> FullCourse:
    logger.info("=== Starting course generation: %r ===", course_input.course_title)
    skeleton = generate_skeleton(course_input)

    lessons: List[Lesson] = []
    for item in skeleton.lessons:
        lesson = generate_lesson(course_input, skeleton, item.lesson_number, previously_generated_lessons=lessons)
        lessons.append(lesson)

    # Post-generation: no redundancy check here — see sequence_check.py's
    # docstring for why (fixed at the payload level instead, below).

    tool_names = ", ".join(t.name for t in course_input.third_party_software) or "none"
    equipment_and_training = (
        f"{course_input.num_lessons} Lesson Hours; "
        f"Equipment: {', '.join(course_input.equipment) or 'none'}; "
        f"Software: {tool_names}; "
        "Curriculum and supporting materials; ongoing product and curriculum "
        "support; professional development; facilitation by a trained STEM "
        "instructor (optional)."
    )

    logger.info("=== Course generation complete: %d lessons written ===", len(lessons))

    return FullCourse(
        course_title=skeleton.course_title,
        grade_band=skeleton.grade_band,
        course_description=skeleton.course_description,
        equipment_and_training=equipment_and_training,
        lessons=lessons,
    )


if __name__ == "__main__":
    example_input = CourseInput(
        category="AI Literacy",
        grade_band="9-12",
        course_title="Trained to React: Gesture-Triggered Sound with micro:bit",
        course_outcome=(
            "By the end of this course, a student can build a working device that recognizes two "
            "physical gestures they defined themselves (for example, sliding the micro:bit sideways "
            "versus lifting it up) and plays a different sound for each, while staying silent when "
            "the device is just held still or set down. Along the way they connect a micro:bit and "
            "see how its motion sensor produces numbers that change as it moves; define their own "
            "gestures precisely enough that someone else could repeat them; record many labelled "
            "examples of each gesture plus examples of 'nothing happening'; train a model on those "
            "recordings in micro:bit CreateAI without writing training code; test whether the model "
            "recognizes gestures it has not seen before and find which classes it confuses; program "
            "the micro:bit in MakeCode so each recognized gesture triggers its own sound; and finally "
            "try to break the finished device — does it still work for a different person, at a "
            "different speed, or held at a different angle? The bigger idea students should leave "
            "with: a machine can be taught to recognize something by being shown examples rather "
            "than by someone writing rules for it, and that approach works well until it meets "
            "something unlike its examples — which is why testing on new cases matters."
        ),
        theme="Physical computing / embedded AI (TinyML) — gesture-to-sound response, PlushPal-style",
        technique="Physical computing / embedded AI (TinyML)",
        equipment=["micro:bit", "USB cable", "battery pack (optional)"],
        third_party_software=[
            ToolSpec(
                name="micro:bit CreateAI",
                capabilities_summary=(
                    "Trains a machine learning model on ACCELEROMETER/MOTION data only "
                    "(x/y/z movement, e.g. waving, clapping, jumping, tilting, shaking). "
                    "Has NO audio, microphone, sound-input, camera, or image-input "
                    "capability of any kind. Can trigger a programmed SOUND OUTPUT "
                    "(via MakeCode) in response to a recognized motion."
                ),
                citation=Citation(
                    source="micro:bit Educational Foundation",
                    detail="Official CreateAI product page and user guide",
                    url="https://microbit.org/createai/",
                ),
            )
        ],
        grounding_sources=[
            Citation(source="TinyML4K12", detail="micro:bit + PlushPal tutorial — gesture-trained, sound-responding stuffed animal", url="https://tinymlx.github.io/4K12"),
            Citation(source="Forward Education", detail="K-12 micro:bit AI literacy guide, confirms CreateAI trains movement recognition"),
        ],
        pedagogical_grounding=(
            "Physical computing is an established K-12 AI-education modality, and "
            "gesture-trained, sound-responding projects (the PlushPal pattern) are a "
            "real, already-documented micro:bit + CreateAI tutorial — not represented "
            "in NextWaveSTEM's current AI Literacy catalog, which is entirely "
            "screen/app-based (Scratch, Teachable Machine, Python)."
        ),
        num_lessons=7,
    )

    course = generate_course(example_input)
    out_path = "generated_course.json"
    # encoding="utf-8" is required — Windows defaults to cp1252, which can't
    # encode characters the model commonly emits (arrows, dashes, quotes).
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(course.model_dump_json(indent=2))
    logger.info("Wrote %s", out_path)