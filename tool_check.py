"""
Deterministic (non-LLM) check for tool-capability mismatches.

Origin: a real failure where the model wrote 4+ lessons instructing students
to record AUDIO using "micro:bit CreateAI" — a tool that only supports
accelerometer/motion data. The tool's capabilities were never actually
verified or passed in; the model filled the gap from memory and got it wrong,
confidently, repeatedly, across the whole course.

This check is cheap and mechanical on purpose: it doesn't understand
meaning, it just flags when a lesson's text pairs a tool's name with a
capability-modality keyword that never appears in that tool's *stated*
capabilities_summary. This is a generalist feature — it works the same way
regardless of category (drones, AI, robotics, anything with named
third-party tools), because it never depends on subject-matter content,
only on the tool/text pairing.

This is intentionally a WARNING-producing check, not a hard raise: keyword
matching has false positives (a tool's summary might describe audio output
without listing it under a mismatched heading, etc.), so results are meant
to be surfaced to the human reviewer (Stage 4), not to silently block
generation on their own.
"""

from __future__ import annotations

import logging
import re
from typing import List

from schema import Lesson, ToolSpec

logger = logging.getLogger(__name__)

MODALITY_KEYWORDS = {
    "audio": ["audio", "sound", "microphone", "mic ", "voice", "speech", "wake word", "recording a sample"],
    "motion": ["accelerometer", "motion", "movement", "gesture", "tilt", "shake"],
    "vision": ["camera", "image", "vision", "photo", "video", "facial recognition"],
}

# A naive "does this keyword appear anywhere in the summary" check gets
# inverted by sentences like "has NO audio ... capability" — the word
# "audio" appears, but the sentence means the opposite of "supported."
# Negation is checked per-clause (split on . and ;) so a keyword is only
# counted as supported if it appears in a clause with no negation word.
NEGATION_WORDS = ["no ", "not ", "n't", "none", "without", "lacks", "lack of", "cannot", "doesn't", "does not"]


def _supported_modalities(tool: ToolSpec) -> set:
    cap = tool.capabilities_summary.lower()
    clauses = re.split(r"[.;]", cap)
    supported = set()
    for modality, keywords in MODALITY_KEYWORDS.items():
        for clause in clauses:
            if any(kw in clause for kw in keywords):
                if not any(neg in clause for neg in NEGATION_WORDS):
                    supported.add(modality)
    return supported


def check_tool_capability_consistency(lesson: Lesson, tools: List[ToolSpec]) -> List[str]:
    """
    Returns a list of human-readable warnings. Empty list means nothing
    flagged (not the same as "verified correct" — this is a heuristic, not
    a proof).
    """
    warnings: List[str] = []
    if not tools:
        return warnings

    text_blobs = [lesson.content, lesson.activity] + [c.text for c in lesson.claims]

    for tool in tools:
        supported = _supported_modalities(tool)
        # A tool with no detected modality keywords in its own summary is a
        # sign the summary itself is too vague to check against — flag that
        # too, since an unverifiable capabilities_summary defeats the point
        # of this schema field.
        if not supported:
            logger.warning(
                "Tool %r has a capabilities_summary with no detectable modality "
                "keywords — too vague to check against: %r",
                tool.name,
                tool.capabilities_summary,
            )

        tool_name_lower = tool.name.lower()
        short_name = tool_name_lower.split(":")[0]  # e.g. "microbit createai" -> "microbit"

        for blob in text_blobs:
            blob_lower = blob.lower()
            if tool_name_lower not in blob_lower and short_name not in blob_lower:
                continue
            for modality, keywords in MODALITY_KEYWORDS.items():
                if modality in supported:
                    continue
                if any(kw in blob_lower for kw in keywords):
                    warning = (
                        f"Lesson {lesson.lesson_number} ('{lesson.title}'): text references "
                        f"'{modality}' capability alongside tool '{tool.name}', but "
                        f"capabilities_summary does not support {modality}. "
                        f"Excerpt: {blob[:150]!r}"
                    )
                    warnings.append(warning)
                    logger.warning(warning)

    return warnings