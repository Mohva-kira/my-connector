from linkedin_market.cv_parcours import format_parcours


def test_one_block_per_experience_with_newlines():
    text = (
        "Founder\nFly Enterprise · Indépendant\noct. 2021 - Présent · 5 ans\nBamako\nJ'ai lancé la société.\n"
        "Lead Developer\nSUPINFO\njanv. 2019 - sept. 2021\nParis\nAPIs."
    )
    out = format_parcours(text).experience
    assert "### Founder — Fly Enterprise · Indépendant" in out
    assert "*10/2021 – Présent*" in out  # format de date lu par les ATS, durée LinkedIn retirée
    assert "### Lead Developer — SUPINFO" in out and "*01/2019 – 09/2021*" in out
    assert out.index("J'ai lancé la société.") < out.index("### Lead Developer")


def test_glued_text_is_split_per_experience():
    text = (
        "Lead DeveloperSUPINFO · CDIjanv. 2019 - sept. 2021 · 2 ans 9 moisParis."
        "Développement d'APIs.CTOAcme · CDI2022 - Présent"
    )
    out = format_parcours(text).experience
    assert out.count("### ") == 2
    assert "*01/2019 – 09/2021*" in out and "*2022 – Présent*" in out


def test_without_period_text_becomes_paragraphs():
    assert format_parcours("a\nb\n\nc").experience == "a\n\nb\n\nc"


def test_tail_sections_are_split_and_mapped():
    """Formation (avec ses propres périodes), Langues… deviennent des sections ; Compétences et Recommandations non."""
    text = (
        "Lead Developer\nSUPINFO\njanv. 2019 - sept. 2021\nAPIs.\n"
        "Formation\nSUPINFO\nMaster, Informatique\n2014 - 2019\n"
        "Langues\nFrançais\nAnglais\nCompétences\nPython\nRecommandations\nJane Doe : très bon."
    )
    parcours = format_parcours(text)
    assert "Master" not in parcours.experience and "Formation" not in parcours.experience
    assert parcours.sections == [
        ("education", "SUPINFO\n\nMaster, Informatique\n\n2014 – 2019"),
        ("languages", "Français\n\nAnglais"),
    ]


def test_glued_tail_sections():
    text = "Lead DeveloperSUPINFOjanv. 2019 - sept. 2021Paris.APIs.FormationSUPINFOMaster2014 - 2019LanguesFrançais"
    parcours = format_parcours(text)
    assert [key for key, _ in parcours.sections] == ["education", "languages"]
    assert "2014 – 2019" in parcours.sections[0][1]
