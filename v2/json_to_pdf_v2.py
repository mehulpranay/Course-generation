"""
Converts a Stage 2 FullCourse JSON output into a readable PDF for human
review -- NOT a teacher-facing deliverable. Purpose: let a domain expert
(or you) eyeball structure, progression, and claim types quickly, to catch
hallucination / causal / sequencing issues before anything is polished.

v2 note: no introduces/prerequisites tags to render -- v2 has no skeleton
step (see schema.py). Otherwise this is the same review tool as v1: every
claim is shown with its type tag ([MECHANISTIC], [HARD_FACT: source],
[CAUSAL], [PEDAGOGICAL]), and a hard_fact claim's retrieved_snippet (the
actual text the model found via search, if captured) is now shown right
under it, so a reviewer can sanity-check the claim against its own source
without leaving the PDF.

The model's `content` and `activity` fields come back as Markdown-ish text
(**bold** headers, "1. " numbered steps, "- " bullets) inside a single
string with no real line breaks. ReportLab's Paragraph does not interpret
any of that -- it just flows it as one block. render_rich_text() below
parses it into separate, properly formatted flowables instead.

Usage:
    python json_to_pdf.py generated_course.json course_review.pdf
"""

from __future__ import annotations

import json
import re
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

CLAIM_COLORS = {
    "mechanistic": "#2563eb",  # blue
    "hard_fact": "#b91c1c",    # red -- most important to scrutinize
    "causal": "#b45309",       # amber
    "pedagogical": "#6b7280",  # gray -- no check needed
}


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LessonHeading", parent=styles["Heading1"], spaceBefore=20, spaceAfter=6))
    styles.add(ParagraphStyle(name="LearningTarget", parent=styles["Italic"], spaceAfter=10))
    styles.add(ParagraphStyle(name="SectionLabel", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name="ClaimText", parent=styles["Normal"], spaceAfter=2, leftIndent=10))
    styles.add(ParagraphStyle(name="SnippetText", parent=styles["Italic"], spaceAfter=6, leftIndent=20))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceAfter=8, leading=14))
    styles.add(ParagraphStyle(name="BodyBullet", parent=styles["Normal"], spaceAfter=4, leading=14, leftIndent=14))
    return styles


def _md_bold_to_reportlab(text: str) -> str:
    """Convert **bold** markdown to ReportLab's <b> tag."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def render_rich_text(raw: str, styles) -> list[Flowable]:
    """
    Parses a single Markdown-ish string (no real newlines, but containing
    **bold** headers, '1. ' numbered steps, and '- ' bullets run together)
    into a list of separate, properly formatted flowables.
    """
    text = raw.strip()

    # Models sometimes emit ESCAPED newlines (a literal backslash-n) rather
    # than real ones -- convert them to real newlines before other parsing.
    text = text.replace("\\n", "\n")

    # Strip markdown ATX headers -- the PDF already has its own section
    # headings, so these would render as literal "###" noise.
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Insert a real line break before each numbered step, but only when it
    # follows sentence-ending punctuation, to avoid splitting on decimals
    # or unrelated numbers mid-sentence.
    text = re.sub(r"(?<=[\.\?\!])\s+(?=\d+\.\s)", "\n", text)
    text = re.sub(r"^(?=\d+\.\s)", "", text)

    # Insert a line break before dash-bullets that precede a bold term.
    text = re.sub(r"\s+-\s+(?=\*\*)", "\n- ", text)

    # Insert a line break before bold headers appearing mid-text.
    text = re.sub(r"(?<=[\.\?\!:])\s+(?=\*\*)", "\n", text)

    text = _md_bold_to_reportlab(text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Models sometimes emit "1." on its own line with the step text on the
    # NEXT line. Join those back together so a step renders as one item.
    joined: list[str] = []
    i = 0
    while i < len(lines):
        if re.fullmatch(r"\d+\.", lines[i]) and i + 1 < len(lines):
            joined.append(f"{lines[i]} {lines[i + 1]}")
            i += 2
        else:
            joined.append(lines[i])
            i += 1
    lines = joined

    flowables: list[Flowable] = []
    for line in lines:
        numbered = re.match(r"^(\d+)\.\s*(.*)", line)
        if numbered:
            body = numbered.group(2).strip()
            label = f"<b>{numbered.group(1)}.</b>"
            flowables.append(Paragraph(f"{label} {body}" if body else label, styles["BodyBullet"]))
        elif line.startswith("- ") or line.startswith("\u2022 "):
            flowables.append(Paragraph(f"\u2022 {line[2:].strip()}", styles["BodyBullet"]))
        else:
            flowables.append(Paragraph(line, styles["Body"]))
    return flowables


def claim_paragraphs(claim: dict, styles) -> list[Flowable]:
    """
    Returns the claim's tag+text paragraph, plus a retrieved-snippet
    paragraph right under it when one was captured during generation.
    """
    ctype = claim["claim_type"]
    color = CLAIM_COLORS.get(ctype, "#000000")
    tag = f"[{ctype.upper()}]"
    citation = claim.get("citation")
    snippet = None
    if ctype == "hard_fact" and citation:
        src = citation.get("source", "unknown source")
        detail = citation.get("detail")
        tag += f" (source: {src}{' -- ' + detail if detail else ''})"
        snippet = citation.get("retrieved_snippet")
    elif ctype == "hard_fact":
        tag += " (!! NO CITATION)"
    text = f'<font color="{color}"><b>{tag}</b></font> {claim["text"]}'
    out = [Paragraph(text, styles["ClaimText"])]
    if snippet:
        out.append(Paragraph(f'<font color="#374151">"{snippet}"</font>', styles["SnippetText"]))
    return out


def build_pdf(course: dict, out_path: str) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(out_path, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = []

    # Title / course-level info
    story.append(Paragraph(course["course_title"], styles["Title"]))
    story.append(Paragraph(f"Grade Band: {course['grade_band']}", styles["Heading3"]))
    story.append(Spacer(1, 8))
    story.extend(render_rich_text(course["course_description"], styles))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Equipment, Curriculum, and Training", styles["SectionLabel"]))
    story.extend(render_rich_text(course["equipment_and_training"], styles))
    story.append(PageBreak())

    for lesson in course["lessons"]:
        story.append(Paragraph(f"Lesson {lesson['lesson_number']}: {lesson['title']}", styles["LessonHeading"]))
        story.append(Paragraph(f"Learning Target: {lesson['learning_target']}", styles["LearningTarget"]))

        story.append(Paragraph("Content", styles["SectionLabel"]))
        story.extend(render_rich_text(lesson["content"], styles))

        story.append(Paragraph("Activity", styles["SectionLabel"]))
        story.extend(render_rich_text(lesson["activity"], styles))

        if lesson.get("equipment_used"):
            story.append(Paragraph("Equipment Used: " + ", ".join(lesson["equipment_used"]), styles["Normal"]))

        if lesson.get("claims"):
            story.append(Paragraph(f"Claims ({len(lesson['claims'])}) -- review these first", styles["SectionLabel"]))
            for claim in lesson["claims"]:
                story.extend(claim_paragraphs(claim, styles))

        if lesson.get("assessment"):
            a = lesson["assessment"]
            story.append(Paragraph("Assessment", styles["SectionLabel"]))
            story.append(Paragraph(a["question"], styles["Normal"]))
            items = []
            for opt in a["options"]:
                marker = " \u2713 CORRECT" if opt == a["correct_answer"] else ""
                items.append(ListItem(Paragraph(opt + marker, styles["Normal"])))
            story.append(ListFlowable(items, bulletType="bullet"))
            if a.get("supporting_claim_index") is not None:
                story.append(
                    Paragraph(f"(Correct answer justified by claim #{a['supporting_claim_index']} above)", styles["Italic"])
                )

        story.append(Spacer(1, 14))

    doc.build(story)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_pdf.py <input.json> <output.pdf>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        course_data = json.load(f)

    build_pdf(course_data, sys.argv[2])
    print(f"Wrote {sys.argv[2]}")