"""CV Markdown -> PDF (fpdf2) pour le téléversement LinkedIn et la pièce jointe email, au format ATS.

Une colonne, texte réel (jamais d'image), coordonnées dans le corps de la page (les ATS ignorent souvent en-tête et
pied de page PDF), liens cliquables dont le texte reste lisible. Police Liberation Sans embarquée (Unicode, métriques
d'Arial, licence OFL : ``assets/fonts``) : « œ », « € » ou « – » sont extraits tels quels par les ATS ; un caractère
absent de la police (émoji…) est retiré plutôt que rendu par un glyphe vide.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from . import cv_format

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_FONT = "cv"
_STYLES = {"": "Regular", "B": "Bold", "I": "Italic", "BI": "BoldItalic"}
_MUTED = (90, 90, 90)
_RULE = (200, 200, 200)


def _fpdf_markdown(text: str) -> str:
    """Segments ``cv_format.spans`` -> balisage fpdf2 (``**gras**``, ``__italique__``, ``[texte](url)``).

    Les marqueurs fpdf2 présents dans le texte brut (``--`` souligne, ``__`` italique) sont neutralisés.
    """
    out = []
    for part, style, url in cv_format.spans(text):
        part = part.replace("--", "–").replace("__", "_").replace("**", "*")
        if url:
            out.append(f"[{part}]({url})")
        elif style == "B":
            out.append(f"**{part}**")
        elif style == "I":
            out.append(f"__{part}__")
        else:
            out.append(part)
    return "".join(out)


def render_pdf(
    markdown: str, title: str = "CV", *, author: str = "", keywords: Iterable[str] = (), lang: str = "fr"
) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    for style, name in _STYLES.items():
        pdf.add_font(_FONT, style, FONT_DIR / f"LiberationSans-{name}.ttf")
    glyphs = pdf.fonts[_FONT].cmap
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 16, 18)
    pdf.set_title(title[:120])
    pdf.set_author(author)
    pdf.set_subject(title[:120])
    pdf.set_keywords(", ".join(keywords))
    pdf.set_creator("linkedin_market")
    pdf.set_lang(lang)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def write(
        text: str,
        size: float,
        style: str = "",
        *,
        indent: float = 0,
        before: float = 0,
        after: float = 1.5,
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        text = "".join(c for c in text if ord(c) in glyphs)
        pdf.set_font(_FONT, style, size)
        pdf.set_text_color(*color)
        pdf.ln(before)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(
            width - indent, size * 0.45 + 1, _fpdf_markdown(text), markdown=True, new_x="LMARGIN", new_y="NEXT"
        )
        pdf.ln(after)

    in_header = True  # lignes entre le nom et la première section : titre et coordonnées
    for block in cv_format.blocks(markdown):
        if block.kind == "blank":
            pdf.ln(1.2)
        elif block.kind == "heading" and block.level == 1:
            write(block.text, 18, "B", after=1)
        elif block.kind == "heading" and block.level == 2:
            in_header = False
            write(block.text.upper(), 11.5, "B", before=3, after=0.8)
            pdf.set_draw_color(*_RULE)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + width, pdf.get_y())
            pdf.ln(1.5)
        elif block.kind == "heading":
            write(block.text, 10.5, "B", before=2, after=0.5)
        elif block.kind == "bullet":
            pdf.set_font(_FONT, "", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.set_x(pdf.l_margin + 2)
            pdf.cell(4, 10 * 0.45 + 1, "•")
            write(block.text, 10, indent=6, after=0.6)
        elif in_header and block.text.startswith("**"):
            write(block.text, 12, after=0.8)
        elif in_header:
            write(block.text, 9.5, after=0.8, color=_MUTED)
        elif block.text.startswith("*") and not block.text.startswith("**"):
            write(block.text, 9.5, after=0.8, color=_MUTED)
        else:
            write(block.text, 10, after=1)
    return bytes(pdf.output())
