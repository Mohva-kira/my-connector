"""Opposition saisie par l'utilisateur (I-13, V4-1, V4-9, V4-10) : ``POST /api/suppressions`` et contacts opposés."""

import pytest
from sqlalchemy import select

from linkedin_market import application, contacts
from linkedin_market.db import models, session
from linkedin_market.db.repositories import suppressions
from linkedin_market.tests.test_message_quota import FakeCtx, client, manager, prepare  # noqa: F401  (fixtures)
from linkedin_market.web import services

pytestmark = pytest.mark.db

PROFILE = "https://www.linkedin.com/in/jane-doe/"
OPPOSED_422 = "Ce contact s'est opposé à tout contact : ses coordonnées ne peuvent plus être saisies."


def hashes() -> dict[str, str]:
    with session.transaction() as s:
        return {r.recipient_hash: r.reason for r in s.execute(select(models.SuppressionEntry)).scalars()}


def contact_rows() -> list[models.CompanyContact]:
    with session.transaction() as s:
        return list(s.execute(select(models.CompanyContact).order_by(models.CompanyContact.full_name)).scalars())


def jane(ref: str) -> None:
    application.update_application(
        ref,
        contacts=[
            {
                "full_name": "Jane Doe",
                "profile_url": PROFILE,
                "email": "Jane.Doe@acme.com",
                "headline": "Talent Lead",
                "source": "linkedin_equipe_recrutement",
            },
            {"full_name": "Autre", "email": "autre@acme.com"},
        ],
    )


def oppose(client, recipient: str, reason: str = "désinscription"):  # noqa: F811
    return client.post("/api/suppressions", json={"recipient": recipient, "reason": reason})


# ------------------------------------------------------------ POST /suppressions
def test_creation_email_url_et_rejeu_idempotent(client):  # noqa: F811
    for recipient in ("jane@acme.com", PROFILE, "jane@acme.com", "  JANE@acme.com  "):
        response = oppose(client, recipient)
        assert response.status_code == 201 and response.json() == {"ok": True}
    assert len(hashes()) == 2


def test_motif_d_origine_conserve_au_rejeu(client):  # noqa: F811
    oppose(client, "jane@acme.com", "plainte")
    oppose(client, "jane@acme.com", "refus_oral")
    assert list(hashes().values()) == ["plainte"]


def test_validation_messages_exacts(client):  # noqa: F811
    bad = oppose(client, "pas-un-mail")
    assert (
        bad.status_code == 422
        and bad.json()["detail"] == "recipient : adresse email ou URL de profil LinkedIn attendue."
    )
    assert oppose(client, "https://evil.example.com/in/x").status_code == 422
    assert oppose(client, "").status_code == 422
    reason = oppose(client, "jane@acme.com", "inconnu")
    assert reason.status_code == 422
    assert reason.json()["detail"] == "reason : doit être l'un de : désinscription, refus_oral, plainte"
    assert hashes() == {}


def test_une_requete_bloque_email_et_url_et_efface_les_deux_champs(client, data_dir):  # noqa: F811
    (ref,) = prepare(data_dir)
    jane(ref)
    assert oppose(client, "jane.doe@acme.com").status_code == 201  # une seule coordonnée fournie
    assert set(hashes()) == suppressions.fingerprints("jane.doe@acme.com", PROFILE)  # V4-1 : email ET profil
    rows = {r.full_name: r for r in contact_rows()}
    assert rows["Jane Doe"].opposed_at is not None and rows["Jane Doe"].email is None
    assert rows["Jane Doe"].profile_url == "" and rows["Jane Doe"].headline == "Talent Lead"
    assert rows["Autre"].opposed_at is None and rows["Autre"].email == "autre@acme.com"
    listed = {c["full_name"]: c for c in client.get(f"/api/applications/{ref}").json()["contacts"]}
    assert listed["Jane Doe"]["opposed"] is True and listed["Jane Doe"]["email"] is None
    assert listed["Jane Doe"]["profile_url"] == "" and listed["Jane Doe"]["collected_at"].endswith("Z")
    assert listed["Autre"]["opposed"] is False
    assert client.get(f"/api/applications/{ref}/message-draft", params={"to": PROFILE}).json()["opposed"] is True


def test_evenement_contacts_sans_donnee_personnelle_et_rejeu(client, data_dir):  # noqa: F811
    (ref,) = prepare(data_dir)
    jane(ref)
    oppose(client, PROFILE)
    events = [e for e in application.get_application(ref)["history"] if e["kind"] == "contacts"]
    assert len(events) == 1
    assert {k: v for k, v in events[0].items() if k not in ("at", "kind")} == {"opposition": True}
    oppose(client, "jane.doe@acme.com")  # rejeu par l'autre coordonnée : contact déjà opposé, aucun nouvel événement
    again = [e for e in application.get_application(ref)["history"] if e["kind"] == "contacts"]
    assert len(again) == 1


def test_aucun_evenement_sans_contact_apparie(client, data_dir):  # noqa: F811
    (ref,) = prepare(data_dir)
    jane(ref)
    oppose(client, "inconnu@acme.com")
    assert [e for e in application.get_application(ref)["history"] if e["kind"] == "contacts"] == []
    assert len(hashes()) == 1


def test_aucune_empreinte_en_clair_en_base(client, data_dir):  # noqa: F811
    (ref,) = prepare(data_dir)
    jane(ref)
    oppose(client, "jane.doe@acme.com")
    for value in hashes():
        assert len(value) == 64 and "@" not in value and "linkedin" not in value
    with session.transaction() as s:
        reasons = s.execute(select(models.SuppressionEntry.reason)).scalars().all()
    assert set(reasons) == {"désinscription"}


def test_appariement_sur_toutes_les_candidatures(client, data_dir):  # noqa: F811
    refs = prepare(data_dir, 2)
    for ref in refs:
        application.update_application(ref, contacts=[{"full_name": "Jane", "email": "jane@acme.com"}])
    oppose(client, "jane@acme.com")
    assert all(r.opposed_at is not None for r in contact_rows())
    for ref in refs:
        assert any(e["kind"] == "contacts" for e in application.get_application(ref)["history"])


# -------------------------------------------------------- PATCH contacts (V4-10)
def test_patch_echo_d_un_contact_oppose_ni_duplique_ni_supprime(client, data_dir):  # noqa: F811
    (ref,) = prepare(data_dir)
    jane(ref)
    oppose(client, "jane.doe@acme.com")
    current = client.get(f"/api/applications/{ref}").json()["contacts"]
    # Le frontend renvoie les lignes telles que lues (opposed/collected_at en extra, ignorés).
    response = client.patch(f"/api/applications/{ref}", json={"contacts": current})
    assert response.status_code == 200
    rows = {r.full_name: r for r in contact_rows()}
    assert len(rows) == 2 and rows["Jane Doe"].opposed_at is not None
    # Contacts retirés par le client : l'opposé survit, l'autre est supprimé.
    response = client.patch(f"/api/applications/{ref}", json={"contacts": []})
    assert response.status_code == 200
    assert [r.full_name for r in contact_rows()] == ["Jane Doe"]
    assert [c["opposed"] for c in response.json()["contacts"]] == [True]


def test_patch_opposed_du_client_ignore(client, data_dir):  # noqa: F811
    (ref,) = prepare(data_dir)
    response = client.patch(
        f"/api/applications/{ref}",
        json={
            "contacts": [{"full_name": "Ana", "email": "ana@acme.com", "opposed": True, "collected_at": "2000-01-01"}]
        },
    )
    contact = response.json()["contacts"][0]
    assert (
        contact["opposed"] is False and contact["email"] == "ana@acme.com" and contact["collected_at"] != "2000-01-01"
    )


def test_patch_coordonnee_opposee_422_sans_ecriture(client, data_dir):  # noqa: F811
    (ref,) = prepare(data_dir)
    jane(ref)
    oppose(client, "jane.doe@acme.com")
    before = [(r.full_name, r.email, r.profile_url, r.opposed_at) for r in contact_rows()]
    for contact in ({"full_name": "Jane", "email": "JANE.DOE@acme.com"}, {"full_name": "Jane", "profile_url": PROFILE}):
        response = client.patch(
            f"/api/applications/{ref}",
            json={"status": "abandonnée", "contacts": [{"full_name": "Nouveau", "email": "n@acme.com"}, contact]},
        )
        assert response.status_code == 422 and response.json()["detail"] == OPPOSED_422
    assert [(r.full_name, r.email, r.profile_url, r.opposed_at) for r in contact_rows()] == before
    assert application.get_application(ref)["status"] == "préparée"  # aucune écriture, statut compris


# ------------------------------------------------------------ lookup_contacts (d)
def test_lookup_ecarte_silencieusement_un_recruteur_oppose(data_dir, monkeypatch, accept_risks):
    accept_risks()
    (ref,) = prepare(data_dir)
    application.record_opposition("jane@acme.com", "désinscription")
    application.record_opposition("https://www.linkedin.com/in/john-roe", "plainte")

    def recruiter(name: str, url: str, email: str | None) -> dict:
        return {
            "full_name": name,
            "profile_url": url,
            "email": email,
            "headline": "",
            "source": "linkedin_equipe_recrutement",
        }

    found = [
        recruiter("Jane Doe", PROFILE, "jane@acme.com"),
        recruiter("John Roe", "https://fr.linkedin.com/in/John-Roe/", None),
        recruiter("Ana Lys", "https://www.linkedin.com/in/ana-lys/", "ana@acme.com"),
    ]

    class Driver:
        def quit(self) -> None:
            pass

    monkeypatch.setattr("linkedin_market.browser.create_driver", Driver)
    monkeypatch.setattr("linkedin_market.session.login", lambda *a, **k: None)
    monkeypatch.setattr(contacts, "find_hiring_team", lambda *a, **k: [dict(c) for c in found])
    result = services.lookup_contacts(FakeCtx(), ref)
    assert result == {"found": 1, "emails": 1}
    saved = application.get_application(ref)["contacts"]
    assert [c["full_name"] for c in saved] == ["Ana Lys"]
