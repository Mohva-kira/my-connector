"""CV Markdown -> PDF (fpdf2, polices de base) pour le téléversement LinkedIn et la pièce jointe email.

Mise en page volontairement simple (titres, puces, paragraphes) : lisible par les ATS. Les polices de
base ne couvrent que le latin-1 ; la ponctuation typographique est ramenée à son équivalent ASCII
et les caractères non supportés (émojis…) sont remplacés par « ? » plutôt que de faire échouer l'export.
"""

from __future__ import annotations

import re
from pathlib import Path

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...",
    " ": " ", "•": "-", "→": "->", "✓": "v", "✗": "x",
})
_INLINE = re.compile(r"(\*\*|__|`|\*(?=\S)|(?<=\S)\*)")


def _latin1(text: str) -> str:
    return text.translate(_TRANSLATE).encode("latin-1", "replace").decode("latin-1")


def _plain(text: str) -> str:
    """Retire le balisage Markdown en ligne (gras, italique, code) et transforme [texte](url) en « texte (url) »."""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return _INLINE.sub("", text)


def render_pdf(markdown: str, title: str = "CV") -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 16, 18)
    pdf.set_title(_latin1(title)[:120])
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def write(text: str, size: float, style: str = "", indent: float = 0, before: float = 0, after: float = 1.5) -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.ln(before)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(width - indent, size * 0.5 + 1.2, _latin1(_plain(text)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(after)

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line.strip():
            pdf.ln(1.5)
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        numbered = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            if level == 1:
                write(heading.group(2), 18, "B", after=2)
            else:
                write(heading.group(2).upper() if level == 2 else heading.group(2), 11.5 if level == 2 else 10.5, "B", before=3, after=1)
                if level == 2:
                    y = pdf.get_y()
                    pdf.set_draw_color(200, 200, 200)
                    pdf.line(pdf.l_margin, y, pdf.l_margin + width, y)
                    pdf.ln(1.5)
        elif bullet:
            write("- " + bullet.group(1), 10, indent=3, after=0.8)
        elif numbered:
            write(f"{numbered.group(1)}. {numbered.group(2)}", 10, indent=3, after=0.8)
        else:
            write(line.strip(), 10, after=1)
    return bytes(pdf.output())


def ensure_cv_pdf(folder: Path, markdown: str, title: str, name: str) -> Path:
    """Génère ``<dossier>/<name>`` s'il manque ou s'il est plus ancien que ``cv.md``."""
    target = folder / name
    source = folder / "cv.md"
    if target.is_file() and (not source.is_file() or target.stat().st_mtime >= source.stat().st_mtime):
        return target
    target.write_bytes(render_pdf(markdown, title))
    return target
