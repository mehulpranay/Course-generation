"""
Deterministic (non-LLM) skeleton-level coherence check.

Origin: Lesson 1 had students deploy a pre-trained model before Lesson 2
taught them what the device even is — a SKELETON-level ordering error.
Passing more context to lesson generation can't fix this: by the time
content is being written, the order is already locked in. Must be caught on
the skeleton itself, before any lesson content is generated.

Category-agnostic: works off self-declared tags (introduces/prerequisites),
never subject-matter content.

(A separate activity-redundancy check was tried and removed — the actual
Lesson 5/6 overreach turned out to be caused by lesson generation never
seeing later lessons' learning targets at all, not by a lack of forward
context. That's fixed directly in generate.py's payload, not by a
post-hoc similarity check.)
"""

from __future__ import annotations

import logging
from typing import List

from schema import CourseSkeleton

logger = logging.getLogger(__name__)


def check_skeleton_prerequisites(skeleton: CourseSkeleton) -> List[str]:
    """
    Walks lessons in order; a lesson's `prerequisites` must all already
    appear in the accumulated `introduces` set from strictly earlier lessons.
    Returns a list of violation strings (empty = no violations found).
    """
    errors: List[str] = []
    introduced_so_far: set = set()

    for item in sorted(skeleton.lessons, key=lambda x: x.lesson_number):
        missing = [p for p in item.prerequisites if p not in introduced_so_far]
        if missing:
            msg = (
                f"Lesson {item.lesson_number} ('{item.title}') assumes prerequisite(s) "
                f"{missing!r} that no earlier lesson's `introduces` list states."
            )
            errors.append(msg)
            logger.error(msg)
        introduced_so_far.update(item.introduces)

    return errors