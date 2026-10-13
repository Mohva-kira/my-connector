"""Source France Travail (vague 5) : API Offres d'emploi v2 sur transport factice, sans réseau ni base."""

import logging
from urllib.parse import parse_qs

import httpx
import pytest

from linkedin_market import config, sources
from linkedin_market.models import SearchRequest
from linkedin_market.sources import SourceFetchError
from linkedin_market.sources.france_travail import FranceTravailSource, department_of

SECRET = "s3cr3t-client-value"
TOKEN = "tok-abcdef-123456"
OFFER = {
    "id": "194ZXYZ",
    "intitule": "Data Engineer",
    "description": "Pipelines Spark et Airflow en télétravail.",
    "dateCreation": "2026-10-05T08:12:45.000Z",
    "lieuTravail": {"libelle": "75 - Paris"},
    "entreprise": {"nom": "Acme"},
    "origineOffre": {"urlOrigine": "https://candidat.francetravail.fr/offres/recherche/detail/194ZXYZ"},
    "competences": [{"libelle": "Python"}, {"libelle": "Apache Spark"}, {"code": "x"}],
}


class FakeClock:
    """Horloge et ``sleep`` factices : dormir fait avancer le temps, sans attendre."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class FakeApi:
    """Serveur factice : les réponses de recherche sont consommées dans l'ordre ; chaque appel est consigné."""

    def __init__(self, clock: FakeClock, searches: list, expires_in: int = 1499) -> None:
        self.clock, self.searches, self.expires_in = clock, list(searches), expires_in
        self.token_calls: list[tuple[float, dict]] = []
        self.search_calls: list[tuple[float, httpx.Request]] = []
        self.tokens_issued = 0
        self.token_status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "entreprise.francetravail.fr":
            self.token_calls.append((self.clock.t, {k: v[0] for k, v in parse_qs(request.content.decode()).items()}))
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": "invalid_client"})
            self.tokens_issued += 1
            return httpx.Response(
                200,
                json={
                    "access_token": f"{TOKEN}-{self.tokens_issued}",
                    "token_type": "Bearer",
                    "expires_in": self.expires_in,
                },
            )
        assert request.extensions["timeout"]["read"] == 10.0  # délai de 10 s sur chaque appel
        self.search_calls.append((self.clock.t, request))
        step = self.searches.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    @property
    def times(self) -> list[float]:
        return sorted([t for t, _ in self.token_calls] + [t for t, _ in self.search_calls])


def ok(rows=None, status=200, **headers) -> httpx.Response:
    return httpx.Response(status, json={"resultats": [OFFER] if rows is None else rows}, headers=headers)


def build(searches: list, **kwargs) -> tuple[FranceTravailSource, FakeApi, FakeClock]:
    clock = FakeClock()
    api = FakeApi(clock, searches, **kwargs)
    source = FranceTravailSource(
        "client-id", SECRET, transport=httpx.MockTransport(api), sleep=clock.sleep, clock=clock.now
    )
    return source, api, clock


def request(**kwargs) -> SearchRequest:
    return SearchRequest(**{"keyword": "Data Engineer", **kwargs})


# --------------------------------------------------------------------- mapping
def test_correspondance_des_champs():
    source, _api, _clock = build([ok()])
    (item,) = source.fetch(request())
    assert item.id == "job:ft-194ZXYZ" and item.kind == "job" and item.source_code == "france_travail"
    assert item.title == "Data Engineer" and item.company == "Acme" and item.location == "75 - Paris"
    assert item.url == "https://candidat.francetravail.fr/offres/recherche/detail/194ZXYZ"
    assert item.posted_at == "2026-10-05" and item.keyword == "Data Engineer"
    assert item.text.startswith("Pipelines Spark et Airflow") and "Compétences : Python, Apache Spark" in item.text


def test_url_de_repli_sans_origine_et_offres_sans_id_ignorees():
    bare = {"id": "A1", "intitule": "Dev", "description": "x"}
    source, *_ = build([ok([bare, {"intitule": "sans id"}, bare])])
    items = source.fetch(request())
    assert [i.id for i in items] == ["job:ft-A1"]  # doublon et offre sans id écartés
    assert items[0].url.endswith("/detail/A1") and items[0].posted_at == ""


# ------------------------------------------------------------------- requête
def test_parametres_de_recherche_et_jeton():
    source, api, _clock = build([ok(**{"Content-Range": "offres 0-0/1234"})])
    source.fetch(request(max_items=15, location="Paris"))
    ((_, form),) = api.token_calls
    assert form == {
        "grant_type": "client_credentials",
        "client_id": "client-id",
        "client_secret": SECRET,
        "scope": "api_offresdemploiv2 o2dsoffre",
    }
    ((_, call),) = api.search_calls
    params = dict(call.url.params)
    assert params == {"motsCles": "Data Engineer", "publieeDepuis": "31", "range": "0-14", "sort": "1"}
    assert call.headers["Authorization"] == f"Bearer {TOKEN}-1" and call.headers["Accept"] == "application/json"


@pytest.mark.parametrize("caution", ["prudent", "normal", "agressif"])
def test_range_suit_le_plafond_de_la_prudence(caution):
    cap = config.CAUTION_LEVELS[caution].max_items
    source, api, _clock = build([ok()])
    source.fetch(request(caution=caution, max_items=cap))
    assert dict(api.search_calls[0][1].url.params)["range"] == f"0-{cap - 1}"


@pytest.mark.parametrize(
    ("location", "department"),
    [("75", "75"), ("2a", "2A"), ("Lyon (69)", "69"), ("974", "974"), ("Paris", None), ("France", None), ("", None),
     ("Paris 75", None), ("2024", None), ("00", None), ("96", None)],
)  # fmt: skip
def test_departement_seulement_pour_un_code_de_departement(location, department):
    assert department_of(location) == department
    source, api, _clock = build([ok()])
    source.fetch(request(location=location))
    params = dict(api.search_calls[0][1].url.params)
    assert params.get("departement") == department


# ------------------------------------------------------------- codes de retour
def test_204_aucune_offre():
    source, *_ = build([httpx.Response(204)])
    assert source.fetch(request()) == []


def test_206_est_un_succes():
    source, *_ = build([ok(status=206, **{"Content-Range": "offres 0-14/300"})])
    assert len(source.fetch(request())) == 1


def test_400_sans_relance():
    source, api, clock = build([httpx.Response(400, json={"message": "range invalide"})])
    with pytest.raises(SourceFetchError, match="paramètres"):
        source.fetch(request())
    assert len(api.search_calls) == 1 and not {1.0, 2.0} & set(clock.sleeps)  # aucune attente de relance


def test_429_avec_retry_after_remplace_l_attente():
    source, api, _clock = build([httpx.Response(429, headers={"Retry-After": "5"}), ok()])
    assert len(source.fetch(request())) == 1
    (t_first, _), (t_second, _) = api.search_calls
    assert t_second - t_first == pytest.approx(5.0)


def test_retry_after_de_30_s_accepte_au_dela_abandon():
    source, api, _clock = build([httpx.Response(429, headers={"Retry-After": "30"}), ok()])
    assert len(source.fetch(request())) == 1
    assert api.search_calls[1][0] - api.search_calls[0][0] == pytest.approx(30.0)

    source, api, clock = build([httpx.Response(429, headers={"Retry-After": "31"}), ok()])
    with pytest.raises(SourceFetchError, match="31 s"):
        source.fetch(request())
    assert len(api.search_calls) == 1  # abandon immédiat, aucun nouvel essai


def test_5xx_puis_succes_avec_attente_de_1_s_puis_2_s():
    source, api, _clock = build([httpx.Response(503), httpx.Response(502), ok()])
    assert len(source.fetch(request())) == 1
    times = [t for t, _ in api.search_calls]
    assert times[1] - times[0] == pytest.approx(1.0) and times[2] - times[1] == pytest.approx(2.0)


def test_trois_echecs_leve_source_fetch_error():
    source, api, _clock = build([httpx.Response(500), httpx.Response(504), httpx.Response(503), ok()])
    with pytest.raises(SourceFetchError, match="3 essais"):
        source.fetch(request())
    assert len(api.search_calls) == 3  # 3 essais au total, jamais un quatrième


def test_erreurs_reseau_relancees():
    source, api, _clock = build([httpx.ConnectError("boom"), httpx.ReadTimeout("lent"), ok()])
    assert len(source.fetch(request())) == 1
    assert len(api.search_calls) == 3


def test_erreur_reseau_persistante():
    source, _api, _clock = build([httpx.ConnectError("boom")] * 3)
    with pytest.raises(SourceFetchError, match="ne répond pas"):
        source.fetch(request())


def test_autre_code_4xx_sans_relance():
    source, api, _clock = build([httpx.Response(403)])
    with pytest.raises(SourceFetchError, match="403"):
        source.fetch(request())
    assert len(api.search_calls) == 1


# ------------------------------------------------------------------- jeton 401
def test_401_renouvelle_le_jeton_puis_une_seule_nouvelle_tentative_non_comptee():
    # 401, puis deux 503 et un succès : le renouvellement n'a pas consommé un des 3 essais.
    source, api, _clock = build([httpx.Response(401), httpx.Response(503), httpx.Response(503), ok()])
    assert len(source.fetch(request())) == 1
    assert api.tokens_issued == 2 and len(api.search_calls) == 4
    assert api.search_calls[0][1].headers["Authorization"].endswith("-1")
    assert api.search_calls[1][1].headers["Authorization"].endswith("-2")


def test_401_persistant_apres_renouvellement():
    source, api, _clock = build([httpx.Response(401), httpx.Response(401), ok()])
    with pytest.raises(SourceFetchError, match="401"):
        source.fetch(request())
    assert api.tokens_issued == 2 and len(api.search_calls) == 2


def test_jeton_refuse():
    source, api, _clock = build([ok()])
    api.token_status = 401
    with pytest.raises(SourceFetchError, match="FRANCE_TRAVAIL_CLIENT_ID"):
        source.fetch(request())
    assert api.search_calls == []


def test_jeton_garde_en_memoire_puis_renouvele_60_s_avant_l_expiration():
    source, api, clock = build([ok(), ok(), ok()], expires_in=600)
    source.fetch(request())
    source.fetch(request())
    assert api.tokens_issued == 1
    clock.t += 600 - 60  # 60 s avant expires_in
    source.fetch(request())
    assert api.tokens_issued == 2


# ------------------------------------------------------------------- cadence
def test_jamais_plus_de_2_appels_par_seconde_jeton_compris():
    source, api, _clock = build([httpx.Response(503), httpx.Response(401), httpx.Response(503), ok(), ok()])
    source.fetch(request())
    source.fetch(request())
    times = api.times
    assert len(times) >= 6
    assert all(b - a >= 0.5 - 1e-9 for a, b in zip(times, times[1:], strict=False))


# ------------------------------------------------------------------- secrets
def test_aucun_secret_ni_jeton_dans_les_logs_et_les_erreurs(caplog):
    caplog.set_level(logging.DEBUG)
    source, api, _clock = build(
        [httpx.Response(503), httpx.Response(401), ok(), httpx.Response(500)] + [httpx.Response(500)] * 2
    )
    source.fetch(request())
    with pytest.raises(SourceFetchError) as raised:
        source.fetch(request())
    logged = caplog.text + str(raised.value)
    assert logged.count("France Travail") >= 1
    for secret in (SECRET, TOKEN, "client_secret", "Bearer"):
        assert secret not in logged


def test_fabrique_sans_identifiants(monkeypatch):
    monkeypatch.delenv("FRANCE_TRAVAIL_CLIENT_ID", raising=False)
    monkeypatch.delenv("FRANCE_TRAVAIL_CLIENT_SECRET", raising=False)
    with pytest.raises(SourceFetchError, match="FRANCE_TRAVAIL_CLIENT_ID"):
        sources.france_travail_source()
    monkeypatch.setenv("FRANCE_TRAVAIL_CLIENT_ID", "id")
    assert config.france_travail_configured() is False
    monkeypatch.setenv("FRANCE_TRAVAIL_CLIENT_SECRET", "secret")
    assert config.france_travail_configured() is True
    sources.france_travail_source().close()


def test_la_reponse_n_est_jamais_une_exception_non_geree_sur_json_invalide():
    source, *_ = build([httpx.Response(200, content=b"<html>")])
    with pytest.raises(SourceFetchError, match="illisible"):
        source.fetch(request())
