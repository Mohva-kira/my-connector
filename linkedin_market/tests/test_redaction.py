"""Rédaction des tiers avant envoi à Claude (vague 6, E-6, I-11)."""

import json

import pytest

from linkedin_market import application, interpreter, nlp
from linkedin_market.models import Item, Profile

P = nlp.PERSON_PLACEHOLDER


@pytest.mark.parametrize(
    "text",
    [
        "Écris à jane.doe@acme.com pour postuler",
        "Contact : (jane+job@sub.acme.co.uk), merci",
        "mailto:recrutement@acme.fr.",
    ],
)
def test_email_remplace(text):
    out = nlp.redact_third_parties(text)
    assert "@" not in out and P in out


def test_email_regle_de_logmask_source_unique():
    # « a@b » (sans point dans le domaine) n'est pas une adresse selon logmask.EMAIL_ADDRESS : inchangé.
    assert nlp.redact_third_parties("mon@serveur local") == "mon@serveur local"


@pytest.mark.parametrize(
    "phone",
    [
        "06 12 34 56 78",
        "06.12.34.56.78",
        "0612345678",
        "+33 6 12 34 56 78",
        "+33612345678",
        "+1 (415) 555-2671",
        "0033 1 23 45 67 89",
    ],
)
def test_telephones(phone):
    assert nlp.redact_third_parties(f"Appelle le {phone} demain") == f"Appelle le {P} demain"


def test_nombres_ordinaires_inchanges():
    text = "TJM 600 € de 2023-2025, 3 à 5 ans, équipe de 120 personnes, version 3.14.1"
    assert nlp.redact_third_parties(text) == text


@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/in/jane-doe/",
        "http://linkedin.com/in/jane-doe",
        "www.linkedin.com/in/jane-doe",
        "linkedin.com/in/jane-doe",
        "https://fr.linkedin.com/in/jane-d%C3%A9",
        "https://ci.linkedin.com/in/jane-doe/details/experience/?x=1",
    ],
)
def test_url_profil_linkedin(url):
    assert nlp.redact_third_parties(f"Voir {url}. Merci") == f"Voir {P}. Merci"


def test_url_offre_linkedin_conservee():
    text = "Offre : https://www.linkedin.com/jobs/view/123/"
    assert nlp.redact_third_parties(text) == text


def test_nom_connu_casse_et_accents():
    out = nlp.redact_third_parties("José Álvarez recrute. Écrire à JOSE ALVAREZ ou jose  alvarez.", ["José Álvarez"])
    assert "lvarez" not in out and out.count(P) == 3


def test_nom_connu_mot_entier_et_noms_courts_ignores():
    text = "Marie Dupont-Martin et Mariette ; Al est là"
    out = nlp.redact_third_parties(text, ["Marie Dupont", "Al", " ", ""])
    assert "Mariette" in out and "Al est là" in out
    assert nlp.redact_third_parties("Léa Ben travaille", ["Ben"]) == f"Léa {P} travaille"
    assert nlp.redact_third_parties("Benoît", ["Ben"]) == "Benoît"


def test_nom_en_unicode_decompose():
    decomposed = "José Alvarez"
    assert nlp.redact_third_parties(f"Chez {decomposed}.", ["José Álvarez"]) == f"Chez {P}."


@pytest.mark.parametrize(
    "text",
    [
        "Jane Doe et des millions d'autres membres utilisent LinkedIn pour trouver des offres",
        "Jean-Marc Martin et 3 autres relations suivent cette page",
        "John Smith and millions of others use LinkedIn to find jobs",
        "Jane Doe and 12 other members follow this",
    ],
)
def test_preuve_sociale(text):
    assert nlp.redact_third_parties(text) == P


@pytest.mark.parametrize(
    "text",
    [
        "12 relations travaillent ici",
        "1 relation travaille ici",
        "3 connections work here",
        "7 relations travaillent chez Acme",
        "5 relations en commun",
        "2 mutual connections",
    ],
)
def test_relations(text):
    assert nlp.redact_third_parties(f"Avant. {text}\nAprès") == f"Avant. {P}\nAprès"


def test_texte_sans_tiers_inchange():
    text = "Développeur Python senior, 5 ans d'expérience en full remote.\nStack : FastAPI, PostgreSQL."
    assert nlp.redact_third_parties(text, ["Jane Doe"]) == text


def test_idempotence_et_entrees_vides():
    text = (
        "jane@acme.com 06 12 34 56 78 https://fr.linkedin.com/in/jane Jane Doe et des millions d'autres "
        "membres. 4 relations travaillent ici. José Álvarez"
    )
    once = nlp.redact_third_parties(text, ["José Álvarez"])
    assert nlp.redact_third_parties(once, ["José Álvarez"]) == once
    assert nlp.redact_third_parties(None) == "" and nlp.redact_third_parties("") == ""
    assert nlp.redact_third_parties("texte", None) == "texte"


def test_ne_leve_jamais():
    assert nlp.redact_third_parties(12345) == "12345"  # type: ignore[arg-type]
    assert nlp.redact_third_parties("Jane", iter(["Jane Doe", None, 5])) == "Jane"  # type: ignore[list-item]


# ---------------------------------------------------------- intégration (faux client)
class FakeClaude:
    calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        FakeClaude.calls.append((system, user))
        return "=== CV ===\nCV\n=== MESSAGE ===\nBonjour\n=== CONSEILS ===\n- ok"


def test_generation_envoie_une_offre_redigee_et_garde_le_profil(monkeypatch):
    FakeClaude.calls = []
    monkeypatch.setattr(interpreter, "ClaudeInterpreter", FakeClaude)
    profile = Profile(
        text="Moi, ancien collègue de Jane Doe (jane.doe@acme.com).", origin="profil.txt", skills=["python"]
    )
    item = Item(
        kind="job",
        id="job:1",
        url="https://www.linkedin.com/jobs/view/1/",
        title="Dev Python",
        text=(
            "Rejoignez-nous ! Contact : jane.doe@acme.com ou +33 6 12 34 56 78, "
            "profil https://www.linkedin.com/in/jane-doe/. Jane Doe, notre recruteuse, vous répondra. "
            "Marc Petit et des millions d'autres membres utilisent LinkedIn. 8 relations travaillent ici."
        ),
        keyword="python",
    )
    job = {"id": "job:1", "title": "Dev Python", "company": "Acme", "url": item.url, "score": 80}
    app = application.generate(
        profile, item, job, {}, "claude", known_names=["Jane Doe"], home_location="Lyon", timezone="Europe/Paris"
    )
    assert app.generator == "claude"
    ((system, user),) = FakeClaude.calls
    offer = user.split("<offre>")[1].split("</offre>")[0]
    for secret in ("jane.doe@acme.com", "Jane Doe", "06 12", "linkedin.com/in", "Marc Petit", "relations"):
        assert secret not in offer
    assert P in offer
    assert "Moi, ancien collègue de Jane Doe (jane.doe@acme.com)." in user  # profile.text jamais rédigé
    assert "Lyon" in system and "Europe/Paris" in system
    resume = json.loads(user.split("<profil_resume>")[1].split("</profil_resume>")[0])
    assert set(resume) == {"titre", "competences", "annees_experience", "seniorite"}  # aucune donnée de contact
