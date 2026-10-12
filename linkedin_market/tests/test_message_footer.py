"""Pied imposé (I-12), assemblage corps + pied (V4-4), empreintes (V4-5) et ``message-draft`` (V4-6)."""

import hashlib

import pytest
from sqlalchemy import update

from linkedin_market import application, outreach, settings
from linkedin_market.db import models, session
from linkedin_market.db.repositories import suppressions
from linkedin_market.tests.test_message_quota import client, manager, prepare  # noqa: F401  (fixtures partagées)
from linkedin_market.web import services

FOOTER_MAIL = (
    "Je vous contacte au sujet de votre offre « Poste 1 » ; votre adresse provient de une saisie manuelle à partir "
    "d'informations publiques. Pour ne plus recevoir de message de ma part, répondez « STOP » ou écrivez à "
    "me@example.com : votre demande sera prise en compte immédiatement."
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------- compose_footer
def test_pied_email_texte_exact_pour_les_deux_origines():
    manuel = outreach.compose_footer("email", "Dev Python", "manuel", "me@example.com")
    assert manuel == (
        "Je vous contacte au sujet de votre offre « Dev Python » ; votre adresse provient de une saisie manuelle à "
        "partir d'informations publiques. Pour ne plus recevoir de message de ma part, répondez « STOP » ou écrivez "
        "à me@example.com : votre demande sera prise en compte immédiatement."
    )
    equipe = outreach.compose_footer("email", "Dev Python", "linkedin_equipe_recrutement", "me@example.com")
    assert "votre adresse provient de l'équipe de recrutement affichée sur l'offre LinkedIn. Pour ne plus" in equipe


def test_pied_linkedin_texte_exact():
    assert outreach.compose_footer("linkedin", "Dev Python", "manuel", "me@example.com") == (
        "Au sujet de votre offre « Dev Python ». Pour ne plus être contacté(e), répondez STOP."
    )


# ------------------------------------------------------- assemblage (V4-4)
FOOT = outreach.compose_footer("linkedin", "Dev", "manuel", "me@example.com")


def test_assemblage_corps_court_sans_troncature():
    note, truncated = outreach.connection_note("Bonjour Jane.  \n", FOOT)
    assert note == "Bonjour Jane.\n\n" + FOOT and truncated is False
    assert outreach.assemble("Bonjour \n", FOOT) == "Bonjour\n\n" + FOOT


def test_assemblage_corps_exactement_a_la_limite_puis_un_de_plus():
    room = outreach.LINKEDIN_NOTE_MAX - 2 - len(FOOT)
    note, truncated = outreach.connection_note("a" * room, FOOT)
    assert len(note) == 300 and truncated is False
    note, truncated = outreach.connection_note("a" * (room + 1), FOOT)
    assert len(note) == 300 and truncated is True and note == "a" * room + "\n\n" + FOOT


def test_assemblage_corps_long_tronque_sans_points_de_suspension_et_unicode_compte_en_caracteres():
    body = "é😀 " * 200  # caractères hors BMP : comptés une fois chacun (len Python)
    note, truncated = outreach.connection_note(body, FOOT)
    assert truncated is True and len(note) <= 300 and note.endswith("\n\n" + FOOT) and "…" not in note
    kept = note[: -len(FOOT) - 2]
    assert kept == kept.rstrip() and body.startswith(kept)


def test_pied_deja_saisi_n_est_pas_double():
    assert outreach.assemble("Bonjour\n\n" + FOOT, FOOT) == "Bonjour\n\n" + FOOT
    note, truncated = outreach.connection_note("Bonjour\n\n" + FOOT, FOOT)
    assert note == "Bonjour\n\n" + FOOT and truncated is False


# ---------------------------------------------------------------- empreintes
def test_empreinte_email_insensible_a_la_casse_et_aux_espaces():
    assert suppressions.fingerprint("  Jane.Doe@Acme.COM ") == sha("jane.doe@acme.com")


def test_empreinte_url_forme_canonique():
    canonical = sha("https://www.linkedin.com/in/x")
    for url in (
        "https://fr.linkedin.com/in/X/?a=b#c",
        "https://www.linkedin.com/in/x/",
        "https://linkedin.com/in/X?utm=1",
        " https://www.linkedin.com/in/x#frag",
    ):
        assert suppressions.fingerprint(url) == canonical, url


def test_empreinte_absente_pour_une_valeur_invalide():
    for value in ("", "   ", None, "pas-un-mail", "https://evil.example.com/in/x", "a@b", "a@b.io, c@d.io"):
        assert suppressions.fingerprint(value) is None, value
    assert suppressions.fingerprints("x", None, "a@b.io") == {sha("a@b.io")}


# ------------------------------------------------------------- expéditeur
@pytest.mark.db
def test_expediteur_smtp_from_adresse_nue_puis_contact_email_puis_repli(monkeypatch):
    monkeypatch.setenv("SMTP_FROM", "Moi Même <me@example.com>")
    assert services.sender_address() == "me@example.com"
    monkeypatch.delenv("SMTP_FROM")
    assert services.sender_address() == "l'expéditeur de ce message"
    settings.save_settings({"contact_email": "contact@example.com"})
    assert services.sender_address() == "contact@example.com"


# --------------------------------------------------------------- message-draft
@pytest.mark.db
def test_draft_sans_destinataire_canal_email_origine_manuelle(client, data_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SMTP_FROM", "me@example.com")
    (ref,) = prepare(data_dir)
    for query in ("", "?to=", "?to=pas-un-mail", "?to=https://evil.example.com/in/x"):
        response = client.get(f"/api/applications/{ref}/message-draft{query}")
        assert response.status_code == 200, query
        draft = response.json()
        assert draft["footer"] == FOOTER_MAIL and draft["origin"] == "manuel" and draft["opposed"] is False
        assert draft["subject"] == "Candidature — Poste 1"
        assert draft["body"] == "Bonjour" and draft["footer"] not in draft["body"]  # corps sans pied


@pytest.mark.db
def test_draft_contact_linkedin_origine_equipe_et_pied_court(client, data_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SMTP_FROM", "me@example.com")
    (ref,) = prepare(data_dir)
    application.update_application(
        ref,
        contacts=[
            {
                "full_name": "Jane Doe",
                "profile_url": "https://www.linkedin.com/in/jane-doe/",
                "email": "jane@acme.com",
                "source": "linkedin_equipe_recrutement",
            }
        ],
    )
    by_url = client.get(
        f"/api/applications/{ref}/message-draft", params={"to": "https://www.linkedin.com/in/jane-doe/"}
    )
    assert by_url.json()["origin"] == "linkedin_equipe_recrutement"
    assert (
        by_url.json()["footer"] == "Au sujet de votre offre « Poste 1 ». Pour ne plus être contacté(e), répondez STOP."
    )
    by_mail = client.get(f"/api/applications/{ref}/message-draft", params={"to": "JANE@acme.com"}).json()
    assert by_mail["origin"] == "linkedin_equipe_recrutement"
    assert "provient de l'équipe de recrutement affichée sur l'offre LinkedIn." in by_mail["footer"]
    inconnu = client.get(f"/api/applications/{ref}/message-draft", params={"to": "autre@acme.com"}).json()
    assert inconnu["origin"] == "manuel"


@pytest.mark.db
def test_draft_titre_tronque_100_email_60_linkedin(client, data_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SMTP_FROM", "me@example.com")
    (ref,) = prepare(data_dir)
    long_title = "T" * 150
    with session.transaction() as s:
        s.execute(update(models.CollectedItem).values(title=long_title))
    email = client.get(f"/api/applications/{ref}/message-draft", params={"to": "a@acme.com"}).json()
    assert f"« {'T' * 99}… »" in email["footer"] and email["subject"] == f"Candidature — {'T' * 99}…"
    linkedin = client.get(
        f"/api/applications/{ref}/message-draft", params={"to": "https://www.linkedin.com/in/x"}
    ).json()
    assert f"« {'T' * 59}… »" in linkedin["footer"]


@pytest.mark.db
def test_draft_signale_l_opposition_sans_erreur(client, data_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SMTP_FROM", "me@example.com")
    (ref,) = prepare(data_dir)
    application.record_opposition("jane@acme.com", "refus_oral")
    response = client.get(f"/api/applications/{ref}/message-draft", params={"to": "Jane@Acme.com"})
    assert response.status_code == 200 and response.json()["opposed"] is True


@pytest.mark.db
def test_draft_candidature_inconnue_404(client):  # noqa: F811
    response = client.get("/api/applications/job:404/message-draft", params={"to": "a@acme.com"})
    assert response.status_code == 404 and response.json()["detail"] == "Aucune candidature préparée pour cette offre."
