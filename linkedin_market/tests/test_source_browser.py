"""Sources Malt et WTTJ (vague 5) : lecture seule des offres sur pages HTML locales et pilote factice, sans réseau."""

import logging
import re
from pathlib import Path

import pytest

from linkedin_market import sources
from linkedin_market.models import SearchRequest
from linkedin_market.sources import base, malt, wttj

FIXTURES = Path(__file__).parent / "fixtures"
SOURCES_DIR = Path(sources.__file__).parent


def html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeDriver:
    """Pilote factice : ne sait que ``get`` / ``current_url`` / ``page_source`` / ``quit`` (aucun sélecteur Selenium).

    ``routes`` associe un motif d'adresse à une page ; ``redirects`` envoie vers une autre adresse (page de connexion).
    ``on_get`` simule l'utilisateur (annulation) ou la plateforme (blocage) au chargement d'une page.
    """

    def __init__(self, routes: dict[str, str], redirects: dict[str, str] | None = None, on_get=None) -> None:
        self.routes, self.redirects, self.on_get = routes, redirects or {}, on_get
        self.current_url, self.page_source = "about:blank", ""
        self.gets: list[str] = []
        self.quit_called = 0

    def get(self, url: str) -> None:
        self.gets.append(url)
        self.current_url = next((target for pattern, target in self.redirects.items() if pattern in url), url)
        self.page_source = next((page for pattern, page in self.routes.items() if pattern in self.current_url), "")
        if self.on_get:
            self.on_get(self, url)

    def quit(self) -> None:
        self.quit_called += 1


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(base, "human_pause", lambda *a, **k: None)


def wttj_driver(**kwargs) -> FakeDriver:
    routes = {"/jobs?query=": html("wttj_search.html"), "/jobs/": html("wttj_job.html")}
    return FakeDriver(routes, **kwargs)


def malt_driver(**kwargs) -> FakeDriver:
    return FakeDriver({"/s/missions": html("malt_search.html"), "/mission/": html("malt_mission.html")}, **kwargs)


def request(**kwargs) -> SearchRequest:
    return SearchRequest(**{"keyword": "Data Engineer", "max_items": 15, **kwargs})


# ------------------------------------------------------------------ extraction
def test_wttj_extrait_les_offres_sans_doublon_ni_profil():
    driver = wttj_driver()
    scraper = wttj.WttjSource(driver, request())
    items = scraper.collect()
    assert [i.id for i in items] == [
        "job:wttj-data-engineer_paris",
        "job:wttj-ml-engineer_lyon",
        "job:wttj-analytics-engineer_remote",
    ]
    first = items[0]
    assert first.kind == "job" and first.source_code == "wttj" and first.keyword == "Data Engineer"
    assert first.title == "Data Engineer" and first.company == "Acme" and first.location == "Paris"
    assert first.url == "https://www.welcometothejungle.com/fr/companies/acme/jobs/data-engineer_paris"
    assert "pipelines Python, Spark et Airflow" in first.text
    assert "ne doit pas apparaître" not in first.text and "Mentions légales" not in first.text  # <main>, sans script
    assert items[1].company == "Globex"  # sans lien d'entreprise reconnu : nom déduit de l'adresse
    assert not any("/profile/" in url for url in driver.gets)  # I-09 : aucun profil n'est ouvert
    assert scraper.blocked is False
    assert driver.gets[0] == "https://www.welcometothejungle.com/fr/jobs?query=Data+Engineer"


def test_malt_extrait_les_missions_sans_profil():
    driver = malt_driver()
    items = malt.MaltSource(driver, request()).collect()
    assert [i.id for i in items] == ["job:malt-abc123", "job:malt-def456"]  # "/mission/search" et profils écartés
    assert all(i.source_code == "malt" and i.kind == "job" for i in items)
    assert items[0].company == "Globex" and items[0].location == "Remote, France"
    assert "migration d'un entrepôt vers Snowflake" in items[0].text
    assert items[1].url == "https://www.malt.fr/mission/def456"  # requête retirée
    assert not any("/profile/" in url for url in driver.gets)


def test_max_items_limite_les_pages_lues():
    driver = wttj_driver()
    items = wttj.WttjSource(driver, request(max_items=2)).collect()
    assert len(items) == 2 and len(driver.gets) == 3  # une recherche + deux offres


# ---------------------------------------------------------------------- blocage
@pytest.mark.parametrize(
    ("cls", "make", "block_page", "redirect"),
    [
        (wttj.WttjSource, wttj_driver, None, "https://www.welcometothejungle.com/fr/signin?next=%2Fjobs"),
        (malt.MaltSource, malt_driver, None, "https://www.malt.fr/login?redirect=%2Fs%2Fmissions"),
        (wttj.WttjSource, wttj_driver, "blocked_challenge.html", None),
        (malt.MaltSource, malt_driver, "blocked_forbidden.html", None),
    ],
)
def test_blocage_a_la_recherche_arrete_la_source(cls, make, block_page, redirect, caplog):
    caplog.set_level(logging.INFO)
    # La page de résultats est une page de connexion / de défi / de refus : arrêt, 0 élément, aucune page ensuite.
    routes = {"": html(block_page)} if block_page else {"": html("blocked_login.html")}
    driver = FakeDriver(routes, redirects={"": redirect} if redirect else None)
    scraper = cls(driver, request())
    assert scraper.collect() == [] and scraper.blocked is True
    assert len(driver.gets) == 1
    assert scraper._get("https://example.org/une-autre-page") is None
    assert len(driver.gets) == 1  # plus aucun get après le blocage (I-06)
    assert "repos de 6 h" in caplog.text


def test_blocage_en_cours_de_collecte_garde_les_offres_deja_lues():
    def challenge_at_second_offer(driver, url):
        if sum("/jobs/" in u for u in driver.gets) == 2:
            driver.page_source = html("blocked_challenge.html")

    driver = wttj_driver(on_get=challenge_at_second_offer)
    scraper = wttj.WttjSource(driver, request())
    items = scraper.collect()
    assert scraper.blocked is True
    assert [i.id for i in items] == ["job:wttj-data-engineer_paris"]  # pris avant le défi
    assert len(driver.gets) == 3  # recherche, offre 1, offre 2 (défi) : rien après


def test_une_longue_offre_qui_parle_de_captcha_n_est_pas_un_blocage():
    long_page = (
        "<html><head><title>Ingénieur sécurité</title></head><body><main><h1>Ingénieur sécurité</h1><p>"
        + "Tu intègres une équipe qui conçoit des captcha et des défis anti-robots. " * 40
        + "</p></main></body></html>"
    )
    driver = FakeDriver({"/jobs?query=": html("wttj_search.html"), "/jobs/": long_page})
    scraper = wttj.WttjSource(driver, request(max_items=1))
    assert len(scraper.collect()) == 1 and scraper.blocked is False


def test_page_non_reconnue_donne_zero_element_et_un_avertissement(caplog):
    caplog.set_level(logging.WARNING)
    driver = FakeDriver({"": html("unknown_page.html")})
    scraper = malt.MaltSource(driver, request())
    assert scraper.collect() == [] and scraper.blocked is False
    assert len(driver.gets) == 1 and "aucune offre reconnue" in caplog.text


# --------------------------------------------------------------------- annulation
class Stop(Exception):
    pass


def test_annulation_avant_chaque_page_conserve_les_offres_deja_lues():
    calls = []

    def check():
        calls.append(1)
        if len(calls) == 3:  # avant la deuxième offre
            raise Stop

    driver = wttj_driver()
    scraper = wttj.WttjSource(driver, request(), check=check)
    with pytest.raises(Stop):
        scraper.collect()
    assert len(driver.gets) == 2  # aucune page chargée après l'annulation
    assert [i.id for i in scraper.partial] == ["job:wttj-data-engineer_paris"]


# ------------------------------------------------------- sélecteurs et lecture seule
def test_aucun_selecteur_interdit_dans_les_sources_navigateur():
    """R-5 : jamais de classe CSS (générée), de ``By.CLASS_NAME`` ni de sélecteur commençant par un point."""
    forbidden = [
        re.compile(r"By\.CLASS_NAME"),
        re.compile(r"""(?:find_elements?|querySelector(?:All)?|CSS_SELECTOR)\([^)]*["']\s*\."""),
        re.compile(r"""["']\.[A-Za-z_-][\w-]*["']\s*[,)\]]"""),  # chaîne « .classe »
        re.compile(r"\bexecute_script\b"),
    ]
    for name in ("base.py", "malt.py", "wttj.py"):
        code = (SOURCES_DIR / name).read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not pattern.search(code), f"{name} : {pattern.pattern}"


def test_les_sources_navigateur_n_ecrivent_rien():
    code = "".join((SOURCES_DIR / name).read_text(encoding="utf-8") for name in ("base.py", "malt.py", "wttj.py"))
    for write in ("send_keys", ".click(", "submit(", "add_cookie", "ActionChains"):
        assert write not in code


def test_aucune_page_de_profil_n_est_extraite():
    page = base.Page('<a href="/fr/profile/x">X</a><a href="/mission/zz9">Mission</a><a href="/in/y">Y</a>')
    driver = FakeDriver({})
    driver.current_url = "https://www.malt.fr/s/missions"
    found = malt.MaltSource(driver, request()).offer_links(page)
    assert [external_id for external_id, _ in found] == ["malt-zz9"]
