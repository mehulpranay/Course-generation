"""
STEMI v2 -- Stage 2 generation, OpenAI Responses API.

Two calls per course, not one, and not one-per-lesson:

  1. generate_course_draft() -- web search ON, no schema constraint. The
     model writes the whole course as free text and searches whenever it
     needs to, as many times as it needs to, at any point while writing
     any lesson. This is NOT "search once up front" -- it's a normal
     agentic loop, the same way the manual frontier-model test courses
     were actually produced.

  2. structure_course() -- schema ON (strict), web search OFF. Takes the
     draft from step 1 and reshapes it into FullCourse. Does no new
     research and is explicitly told not to invent anything the draft
     doesn't already contain.

How this actually runs, mechanically: `web_search` is a HOSTED tool --
when the model decides to search, OpenAI's own server runs it and hands
the results back to the model automatically, inside the same API call.
Our code never sees that round-trip and never manually feeds a result
back in. generate_course_draft() makes one HTTP request and gets back one
response, even though the model may have searched zero, five, or fifteen
times on OpenAI's side while writing -- before lesson 1, again mid-lesson-9,
however often it needs to. Call 1 runs start to finish and returns only
once the ENTIRE course is fully written; there's no "pause after a
search, resume in call 2" handoff. (That pause-and-resume pattern is real
for CLIENT-side tools, where OpenAI can't run the tool itself and has to
hand it back to us to execute -- web_search isn't that kind of tool.) So
structure_course() isn't reshaping raw search results, and the split
isn't about the search tool's output being the wrong shape either -- it's
given the complete, already-written, already-cited course text and asked
to re-express it as JSON.

Why split into two calls at all, then: OpenAI's Responses API has a
documented, reproducible failure mode when `web_search` and strict
structured output are both enabled on the SAME call -- intermittent
truncated/malformed JSON returned with status "completed", not a clean
error, and it shows up even at the very end, after all searching is
already done. Multiple independent reports show the failure rate drops to
~0 when either the tool or the schema is removed, and holds steady with
both present. It's not that any single call's output is malformed by
nature -- it's that having both capabilities active in one call is itself
the trigger. Splitting into two calls means those two capabilities are
never both active at once, rather than betting a whole-course generation
on a retry loop catching the failure after the fact.

Because step 2 has no search access, step 1 MUST leave a record of every
source it used, in the text itself -- step 2 can only structure what's
already there. merge_search_annotations() does this mechanically: the API
attaches url_citation annotations to specific spans of the model's own
output; this walks those and writes an explicit
"(source: <url> -- title)" marker right after the text it applies to, so
the citation survives being handed to a call that can't search for it
again.

TODO before running: DRAFT_MODEL / STRUCTURE_MODEL below are placeholders.
GPT-6 Astra just launched on a gated rollout and may not be on your
account yet -- confirm what you actually have access to in the OpenAI
console and set these accordingly.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from openai import OpenAI

from schema import CourseInput, FullCourse, ToolSpec

logger = logging.getLogger(__name__)

# TODO: confirm against your OpenAI console before running.
DRAFT_MODEL = "gpt-5.1"
# Structuring is a mechanical reshaping task, not creative writing -- a
# cheaper/faster model may do fine here even if DRAFT_MODEL is the
# expensive one. Worth testing once you know what's actually available.
STRUCTURE_MODEL = "gpt-5.1"

client = OpenAI()


def _tool_context_block(tools: list[ToolSpec]) -> str:
    """
    Formats each tool's starting-point summary for the draft prompt, with
    an explicit instruction that it's a prior to verify, not settled fact.
    """
    if not tools:
        return ""
    lines = ["Tools referenced in this course (verify and update via search -- do not trust blindly):"]
    for t in tools:
        lines.append(f"- {t.name}: {t.capabilities_summary}")
    return "\n".join(lines)


def _build_draft_prompt(course_input: CourseInput) -> str:
    return f"""You are writing a complete {course_input.num_lessons}-lesson course for NextWaveSTEM,
a company selling turnkey STEM/CTE programs to US school districts. The
teacher receiving this is often a generalist, not a subject specialist.

Category: {course_input.category}
Grade band: {course_input.grade_band}
Course title: {course_input.course_title}
Theme: {course_input.theme}
Technique: {course_input.technique}
Equipment available: {", ".join(course_input.equipment) or "none specified"}

Course outcome -- what a student can do at the end:
{course_input.course_outcome}

Why this fits this grade band:
{course_input.pedagogical_grounding}

{_tool_context_block(course_input.third_party_software)}

Use web search whenever you need to check a fact, a specific tool's
capabilities, a specific procedure, or anything else you are not certain
of -- at any point while writing, not just before you start. Search again
mid-course if a later lesson needs something you haven't already
verified.

Every time you state a specific factual, mechanistic, or product-
capability claim, write the source right next to it in the text -- the
URL and, if you can, the exact sentence you're basing the claim on. You
will not have search access when this draft gets reshaped into its final
form, so anything you don't write down here is lost for good.

Write the full course now: a short course description, then all
{course_input.num_lessons} lessons in order, each with a title, a
learning target, teaching content, a hands-on activity with concrete
steps, and one assessment question (3-4 options, correct answer marked).
Use clear section breaks between lessons so the structure is easy to
follow."""


def merge_search_annotations(output) -> str:
    """
    Walks an OpenAI Responses API output list and, for every url_citation
    annotation, appends an inline "(source: url -- title)" marker right
    after the span it applies to. Returns the enriched text.

    If the API's annotation format changes, this degrades to returning an
    empty string (caller falls back to response.output_text) rather than
    raising -- losing inline markers is a quality problem step 2 can
    partly route around by reading the prose itself; crashing here would
    lose the whole draft.
    """
    try:
        pieces: list[str] = []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for block in getattr(item, "content", []):
                text = getattr(block, "text", "")
                annotations = getattr(block, "annotations", []) or []
                if not annotations:
                    pieces.append(text)
                    continue
                # Insert markers back-to-front so earlier offsets stay valid.
                enriched = text
                for ann in sorted(annotations, key=lambda a: getattr(a, "end_index", 0) or 0, reverse=True):
                    if getattr(ann, "type", None) != "url_citation":
                        continue
                    end = getattr(ann, "end_index", None)
                    if end is None:
                        continue
                    url = getattr(ann, "url", "")
                    title = getattr(ann, "title", "")
                    marker = f' (source: {url}{" -- " + title if title else ""})'
                    enriched = enriched[:end] + marker + enriched[end:]
                pieces.append(enriched)
        return "\n".join(pieces) if pieces else ""
    except Exception:
        logger.warning(
            "Failed to merge search annotations into draft text; falling back to plain output_text.",
            exc_info=True,
        )
        return ""


def generate_course_draft(course_input: CourseInput) -> str:
    """Call 1: free-text course, web search on, no schema."""
    response = client.responses.create(
        model=DRAFT_MODEL,
        input=[{"role": "user", "content": _build_draft_prompt(course_input)}],
        tools=[{"type": "web_search"}],
    )
    enriched = merge_search_annotations(response.output)
    return enriched or response.output_text


def _build_structuring_prompt(draft_text: str, course_input: CourseInput) -> str:
    return f"""Reshape the course draft below into the exact schema provided. This is
a structuring task, not a writing task:

- Do not add any fact, claim, or detail that isn't already in the draft.
- Every "(source: url -- ...)" marker in the draft is a real citation the
  model found via search while writing. Carry each one into the matching
  claim's citation field exactly as given -- the url into `url`, the
  quoted or paraphrased basis into `retrieved_snippet` if there's enough
  to go on, and a short source name into `source`. Do not invent a
  citation for a claim that has no marker; if a hard_fact claim has no
  source marker, still tag it hard_fact but leave citation absent rather
  than fabricate one.
- Split the content into individual claims and tag each: mechanistic (how
  something works, no citation needed), hard_fact (a specific product/
  tool/study fact -- citation required if the draft provided one),
  causal (an "A because B" claim), or pedagogical (framing/narrative,
  not a checkable claim).
- num_lessons should equal {course_input.num_lessons}.

--- DRAFT ---
{draft_text}
--- END DRAFT ---"""


def structure_course(draft_text: str, course_input: CourseInput, max_retries: int = 2) -> FullCourse:
    """
    Call 2: schema on (strict), web search off. Retries on any failure --
    malformed output, a refusal, or a transient API error all land here,
    since the fix for all three at this stage is the same: try again.
    """
    prompt = _build_structuring_prompt(draft_text, course_input)
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.parse(
                model=STRUCTURE_MODEL,
                input=[{"role": "user", "content": prompt}],
                text_format=FullCourse,
            )
            return response.output_parsed
        except Exception as exc:
            last_error = exc
            logger.warning("structure_course attempt %d/%d failed: %s", attempt + 1, max_retries + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"structure_course failed after {max_retries + 1} attempts") from last_error


def generate_course(course_input: CourseInput) -> FullCourse:
    """Top-level: draft, then structure."""
    logger.info("Generating draft for %r (%d lessons)...", course_input.course_title, course_input.num_lessons)
    draft = generate_course_draft(course_input)
    logger.info("Draft complete (%d chars). Structuring...", len(draft))
    course = structure_course(draft, course_input)
    logger.info("Structured course complete: %d lessons.", len(course.lessons))
    return course


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python generation.py <course_input.json> <output.json>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        parsed_input = CourseInput.model_validate_json(f.read())

    result = generate_course(parsed_input)

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    print(f"Wrote {sys.argv[2]}")