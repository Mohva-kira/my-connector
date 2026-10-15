"""Format ATS du CV : en-tête, dates, contrôle de conformité, exports PDF et Word lisibles par un ATS."""

import io

import pytest
from docx import Document
from pypdf import PdfReader

from linkedin_market import cv_export, cv_format, cv_pdf, settings

CV = """# Data Engineer

## Profil

8 ans d'expérience : **Python**, Spark. Cœur de métier : pipelines -- fiables 😀.

## Compétences

- **Langages** : Python, SQL

## Expérience professionnelle

### Lead Data Engineer — Acme

*Paris · 10/2021 – Présent*

- Conçu un lakehouse sur AWS

## Formation

### Master — SUPINFO

## Langues

Français, anglais
"""
ADA = cv_format.Identity(
    full_name="Ada Lovelace",
    email="ada@example.com",
    phone="+33 6 12 34 56 78",
    linkedin_url="https://www.linkedin.com/in/ada/",
    location="Lyon",
)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("oct. 2021", "Présent", "10/2021 – Présent"),
        ("janv. 2019", "sept. 2021", "01/2019 – 09/2021"),
        ("août 2020", "déc. 2020", "08/2020 – 12/2020"),
        ("2014", "2019", "2014 – 2019"),
        ("Mar 2020", "present", "03/2020 – Present"),
    ],
)
def test_normalize_period(start, end, expected):
    assert cv_format.normalize_period(start, end) == expected


def test_apply_identity_puts_name_title_and_contact_first():
    out = cv_format.apply_identity("# Data Engineer\n**Candidature : X**\n\n## Profil\n\nTexte", ADA)
    lines = out.splitlines()
    assert lines[:3] == [
        "# Ada Lovelace",
        "**Data Engineer**",
        "Lyon · ada@example.com · +33 6 12 34 56 78 · [linkedin.com/in/ada](https://www.linkedin.com/in/ada)",
    ]
    assert "**Candidature : X**" in lines and out.index("## Profil") > out.index("ada@example.com")


def test_apply_identity_without_settings_keeps_title():
    assert cv_format.apply_identity("# Data Engineer\n\n## Profil\n\nTexte", cv_format.Identity()).startswith(
        "# Data Engineer\n\n## Profil"
    )


def test_validate_cv():
    assert cv_format.validate_cv(cv_format.apply_identity(CV, ADA)) == []
    issues = cv_format.validate_cv("# Dev\n\n## Atouts pour cette offre\n\n| a | b |\n\n![x](y.png)\n")
    assert any("Coordonnées absentes" in i for i in issues)
    assert any("« Atouts pour cette offre »" in i for i in issues)
    assert sum("Section manquante" in i for i in issues) == 3
    assert any("Tableau" in i for i in issues) and any("Image" in i for i in issues)


def test_validate_cv_accepts_english_headings():
    cv = "# Dev\nada@example.com\n\n## Summary\n\nx\n\n## Skills\n\nx\n\n## Professional Experience\n\nx\n"
    assert cv_format.validate_cv(cv) == []


def test_language_of():
    assert cv_format.language_of("We are looking for a data engineer with experience in our team and you will") == "en"
    assert cv_format.language_of("Nous recherchons un développeur pour notre équipe et vous serez") == "fr"


def test_file_stem_is_ascii():
    assert cv_format.file_stem("Adèle Ølsen", "Data Engineer (H/F)") == "Adele-lsen-CV-Data-Engineer-H-F"
    assert cv_format.file_stem("", "Dev") == "CV-Dev"


def test_pdf_text_is_extractable_in_reading_order():
    """Ce que lit un ATS : texte réel, Unicode intact, coordonnées en page 1, sections dans l'ordre."""
    pdf = cv_pdf.render_pdf(cv_format.apply_identity(CV, ADA), "Data Engineer", author="Ada Lovelace")
    reader = PdfReader(io.BytesIO(pdf))
    text = reader.pages[0].extract_text()
    assert text.startswith("Ada Lovelace\nData Engineer\nLyon · ada@example.com")
    assert "linkedin.com/in/ada" in text and "Cœur de métier" in text and "–" in text
    assert "?" not in text and "😀" not in text  # caractère hors police retiré, jamais remplacé par « ? »
    order = [text.index(h) for h in ("PROFIL", "COMPÉTENCES", "EXPÉRIENCE PROFESSIONNELLE", "FORMATION", "LANGUES")]
    assert order == sorted(order)
    assert reader.metadata.author == "Ada Lovelace" and reader.metadata.title == "Data Engineer"
    assert any(a.get_object()["/A"]["/URI"] == "https://www.linkedin.com/in/ada" for a in reader.pages[0]["/Annots"])


def test_docx_uses_native_word_styles(tmp_path):
    (tmp_path / "cv.md").write_text(CV, encoding="utf-8")
    path = cv_export.ensure(tmp_path, cv_format.apply_identity(CV, ADA), "docx", title="Data Engineer", full_name="Ada")
    doc = Document(str(path))
    styles = [(p.style.name, p.text) for p in doc.paragraphs]
    assert styles[0] == ("Heading 1", "Ada Lovelace")
    assert ("Heading 2", "EXPÉRIENCE PROFESSIONNELLE") in styles
    assert ("List Bullet", "Conçu un lakehouse sur AWS") in styles
    assert doc.core_properties.author == "Ada"


def test_export_is_named_cached_and_replaced(tmp_path):
    (tmp_path / "cv.md").write_text(CV, encoding="utf-8")
    first = cv_export.ensure(tmp_path, CV, "pdf", title="Data Engineer", full_name="Ada Lovelace")
    assert first.name == "Ada-Lovelace-CV-Data-Engineer.pdf"
    assert cv_export.ensure(tmp_path, CV, "pdf", title="Data Engineer", full_name="Ada Lovelace") == first
    renamed = cv_export.ensure(tmp_path, CV, "pdf", title="Data Engineer", full_name="Ada King")
    assert [p.name for p in tmp_path.glob("*.pdf")] == [renamed.name]
    cv_export.remove(tmp_path)
    assert list(tmp_path.glob("*.pdf")) == []


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"full_name": "Ada\nLovelace"}, "full_name ne doit contenir ni saut de ligne"),
        ({"full_name": "x" * 81}, "full_name ne peut pas dépasser 80"),
        ({"linkedin_url": "https://example.com/in/ada"}, "linkedin_url doit être l'URL de ton profil"),
    ],
)
def test_identity_settings_are_validated(update, error):
    with pytest.raises(settings.SettingsError, match=error):
        settings.validate_settings(update, current={})


def test_identity_settings_are_normalized():
    clean = settings.validate_settings(
        {"full_name": "  Ada   Lovelace ", "linkedin_url": " https://fr.linkedin.com/in/ada "}, current={}
    )
    assert clean == {"full_name": "Ada Lovelace", "linkedin_url": "https://fr.linkedin.com/in/ada"}
