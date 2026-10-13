# ruff: noqa: F811
"""Scores obsolètes quand le lieu ou le fuseau changent (E-18) ; rapports anciens toujours lisibles."""

from linkedin_market.web import services

from .test_api import client, write_report  # noqa: F401


class _Ctx:
    def log(self, message: str) -> None:
        pass

    def check(self) -> None:
        pass


def _outdated(client) -> bool:
    return client.get("/api/profile").json()["scores_outdated"]


def _analysed(client) -> None:
    """Rapport produit par l'analyse réelle, donc porteur de `scoring_context`."""
    write_report()
    client.put("/api/settings", json={"home_location": "Lyon", "timezone": "Europe/Paris"})
    services.rescore(_Ctx())


def test_scoring_context_enregistre_dans_le_rapport(client):
    _analysed(client)
    assert services.latest_results()["scoring_context"] == {"home_location": "Lyon", "timezone": "Europe/Paris"}
    assert "home_location" not in services.latest_results()["profile"]
    assert _outdated(client) is False


def test_changement_de_home_location_rend_obsolete(client):
    _analysed(client)
    client.put("/api/settings", json={"home_location": "Lyon"})
    assert _outdated(client) is False
    client.put("/api/settings", json={"home_location": "Nantes"})
    assert _outdated(client) is True
    client.put("/api/settings", json={"home_location": "Lyon"})
    assert _outdated(client) is False


def test_changement_de_timezone_rend_obsolete(client):
    _analysed(client)
    client.put("/api/settings", json={"timezone": "Europe/London"})
    assert _outdated(client) is True


def test_autre_cle_seule_ne_change_rien(client):
    _analysed(client)
    client.put("/api/settings", json={"apply_daily_max": 3})
    assert _outdated(client) is False


def test_rescore_remet_a_faux(client):
    _analysed(client)
    client.put("/api/settings", json={"home_location": "Nantes"})
    assert _outdated(client) is True
    services.rescore(_Ctx())
    assert _outdated(client) is False


def test_rapport_ancien_sans_contexte_reste_lisible_et_non_obsolete(client):
    name = write_report()
    client.put("/api/settings", json={"home_location": "Nantes", "timezone": "Europe/London"})
    assert "scoring_context" not in services.latest_results()
    assert client.get(f"/api/reports/{name}").status_code == 200
    assert _outdated(client) is False
