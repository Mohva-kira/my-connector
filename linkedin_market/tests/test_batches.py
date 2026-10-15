"""Collecte par lots de 15 (fenêtre 60 j, 100 offres) : enregistrement et scoring pendant la pause de 2 min."""

import re

import pytest

from linkedin_market import config, pipeline, scraper, storage, trends
from linkedin_market.models import Item, SearchRequest
from linkedin_market.pipeline import BatchRunner


class Clock:
    """Horloge factice : ``tick`` avance le temps au lieu de dormir."""

    def __init__(self):
        self.now = 0.0
        self.ticks = 0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds
        self.ticks += 1


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(pipeline, "_clock", clock)
    monkeypatch.setattr(pipeline, "_tick", clock.tick)
    return clock


def items(count: int, start: int = 0) -> list[Item]:
    return [
        Item(kind="job", id=f"job:{start + n}", url="u", title="t", text="x", keyword="k", scraped_at="2026-10-15")
        for n in range(count)
    ]


def test_valeurs_par_defaut():
    assert config.TREND_WINDOW_DAYS == 60
    for level in config.CAUTION_LEVELS.values():
        assert (level.max_items, level.batch_size, level.batch_pause) == (100, 15, 120.0)
    assert SearchRequest(keyword="x").max_items == 100


def test_fenetre_de_deux_mois_dans_la_recherche():
    assert config.TREND_WINDOW_DAYS * 86400 == 5_184_000
    assert len(trends.weekly_series([], [])["weeks"]) == 9


def test_pause_de_deux_minutes_entre_lots(monkeypatch, clock):
    saved: list[int] = []
    monkeypatch.setattr(storage, "append_items", lambda batch: saved.append(len(batch)))
    runner = BatchRunner(config.CAUTION_LEVELS["prudent"], lambda: None, score=lambda batch: "ok")
    runner(items(15), False)
    assert clock.now >= 120 and saved == [15]
    before = clock.now
    runner(items(5, 15), True)  # dernier lot : on n'attend que la fin du traitement
    assert clock.now == before and saved == [15, 5] and runner.saved == 20


def test_traitement_plus_long_que_la_pause(monkeypatch, clock):
    import threading

    release = threading.Event()
    monkeypatch.setattr(storage, "append_items", lambda batch: release.wait(5))
    levels = config.Caution("t", 100, 0.0, 0.0, batch_pause=0.0)
    runner = BatchRunner(levels, lambda: None)
    threading.Timer(0.3, release.set).start()
    runner(items(15), False)  # la pause est nulle mais on attend la fin de l'enregistrement
    assert release.is_set() and runner.saved == 15


def test_annulation_pendant_la_pause_enregistre_le_lot(monkeypatch, clock):
    saved: list[int] = []
    monkeypatch.setattr(storage, "append_items", lambda batch: saved.append(len(batch)))

    def check():
        if clock.ticks >= 3:
            raise pipeline.Cancelled("Collecte annulée.")

    runner = BatchRunner(config.CAUTION_LEVELS["prudent"], check)
    with pytest.raises(pipeline.Cancelled):
        runner(items(15), False)
    assert saved == [15]


def test_echec_d_enregistrement_arrete_la_collecte(monkeypatch, clock):
    def boom(batch):
        raise OSError("disque plein")

    monkeypatch.setattr(storage, "append_items", boom)
    with pytest.raises(RuntimeError, match="Enregistrement du lot impossible"):
        BatchRunner(config.CAUTION_LEVELS["prudent"], lambda: None)(items(3), True)


def test_echec_de_scoring_non_bloquant(monkeypatch, clock):
    monkeypatch.setattr(storage, "append_items", lambda batch: None)

    def bad_score(batch):
        raise ValueError("x")

    runner = BatchRunner(config.CAUTION_LEVELS["prudent"], lambda: None, score=bad_score)
    runner(items(3), True)
    assert runner.saved == 3


class FakeDriver:
    def __init__(self, count: int):
        self.job_ids = list(range(4000001, 4000001 + count))
        self.current_url, self.title = "about:blank", ""

    def get(self, url: str) -> None:
        self.current_url = url
        match = re.search(r"/jobs/view/(\d+)/", url)
        self.title = f"Data Engineer {match.group(1)} | Acme | LinkedIn" if match else ""

    @property
    def page_source(self) -> str:
        if "/jobs/search/" in self.current_url and "start=0" in self.current_url:
            return " ".join(f'<a href="/jobs/view/{j}/">' for j in self.job_ids)
        return ""

    def execute_script(self, script, *args):
        return "Python Spark" if script is scraper._JS_TEXT else None


def test_scraper_decoupe_les_offres_en_lots_de_15(monkeypatch):
    monkeypatch.setattr(config, "CAUTION_LEVELS", {n: config.Caution(n, 100, 0.0, 0.0) for n in config.CAUTION_LEVELS})
    calls: list[tuple[int, bool]] = []
    request = SearchRequest(keyword="python", max_items=100)
    page = scraper.LinkedInScraper(FakeDriver(100), request, on_batch=lambda b, last: calls.append((len(b), last)))
    result = page.jobs()
    assert len(result) == 100
    assert calls == [(15, False)] * 6 + [(10, True)]
    assert page.partial == []


def test_scraper_sans_on_batch_garde_partial(monkeypatch):
    monkeypatch.setattr(config, "CAUTION_LEVELS", {n: config.Caution(n, 100, 0.0, 0.0) for n in config.CAUTION_LEVELS})
    page = scraper.LinkedInScraper(FakeDriver(20), SearchRequest(keyword="python", max_items=20))
    assert len(page.jobs()) == 20 and len(page.partial) == 20


def test_etat_structure_du_lot_pour_l_indicateur(monkeypatch, clock):
    monkeypatch.setattr(storage, "append_items", lambda batch: None)
    states: list[dict] = []
    runner = BatchRunner(config.CAUTION_LEVELS["prudent"], lambda: None, on_state=states.append, expected=100)
    runner(items(15), False)
    runner(items(5, 15), True)
    assert [s["state"] for s in states] == ["traitement", "reprise", "traitement", "terminé"]
    assert states[0]["batches"] == 7 and states[0]["pause_until"] and states[0]["pause_seconds"] == 120.0
    assert states[2]["pause_until"] is None and states[3]["read"] == 20


def test_batch_scorer_transmet_les_offres_scorees(monkeypatch):
    seen: list[dict] = []
    prof = pipeline.Profile(
        title="t", skills=["python"], years_experience=3, seniority="senior", text="x", origin="test"
    )
    monkeypatch.setattr(pipeline, "build_profile", lambda request: prof)
    monkeypatch.setattr(pipeline.matching.MatchingModel, "load", classmethod(lambda cls: cls()))
    score = pipeline.batch_scorer(SearchRequest(keyword="python"), "", "", on_scored=seen.extend)
    message = score(items(2))
    assert "scorées" in message and {row["id"] for row in seen} == {"job:0", "job:1"}


def test_vue_progressive_des_offres(monkeypatch):
    from linkedin_market.web import services

    monkeypatch.setattr(services.storage, "load_labels", lambda: {})
    monkeypatch.setattr(services.application, "load_applications", lambda: {})

    services.live_start("python")
    try:
        services.live_add([{"id": "job:9", "title": "t", "company": "c", "score": 80.0, "source_code": "linkedin"}])
        rows = services.jobs_overview(None, services.live_snapshot())
        assert [(r["id"], r["provisional"]) for r in rows] == [("job:9", True)]
    finally:
        services.live_stop()
    assert services.live_snapshot() is None
