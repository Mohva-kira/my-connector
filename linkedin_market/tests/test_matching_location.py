"""Compatibilité géographique du matching pilotée par les réglages (E-18) : aucun lieu codé en dur."""

import inspect

import pytest

from linkedin_market import matching, pipeline
from linkedin_market.models import Item, Profile

PROFILE = Profile(text="Python Spark", origin="test", title="Data Engineer", skills=["python"])


def _job(location: str, text: str = "Mission data en Python.") -> Item:
    return Item(
        kind="job", id="job:1", url="https://x/1", title="Data Engineer", text=text, keyword="data", location=location
    )


def _location(job: Item, home: str = "", zone: str = "") -> float:
    features = matching.build_features([job], PROFILE, "data", home, zone)
    return float(features[0].values[matching.FEATURES.index("localisation_compatible")])


def test_lieu_defini_donne_le_bonus():
    assert _location(_job("Lyon, France"), home="Lyon") == 1.0
    assert _location(_job("Lyon, France"), home="Lyon, France") == 1.0
    assert _location(_job("Abidjan"), home="Abidjan, Côte d'Ivoire") == 1.0
    assert _location(_job("Nantes"), home="Lyon") == 0.5


def test_lieu_region_choisie_par_l_utilisateur_donne_le_bonus():
    assert _location(_job("Afrique"), home="Afrique") == 1.0
    assert _location(_job("Abidjan"), home="Abidjan") == 1.0


def test_offre_afrique_sans_lieu_defini_ne_donne_plus_le_bonus():
    assert _location(_job("Afrique"), home="") == 0.5
    assert _location(_job("Remote Africa"), home="") == 1.0  # « remote » reste une mention générique
    assert _location(_job("Africa"), home="") == 0.5


def test_lieu_vide_est_neutre():
    assert _location(_job("Abidjan, Côte d'Ivoire")) == 0.5
    assert _location(_job("Afrique")) == 0.5
    assert _location(_job("Lyon")) == 0.5
    assert _location(_job("Nantes"), home="   ") == 0.5


def test_mentions_generiques_et_bloqueurs_conserves():
    for text in ("Remote", "Worldwide", "Poste international", "EMEA"):
        assert _location(_job(text)) == 1.0
    assert _location(_job("Lyon", "Mission sur site, hybride."), home="Lyon") == 0.0


def test_fuseau_donne_la_ville_sans_region():
    assert _location(_job("Paris"), zone="Europe/Paris") == 1.0
    assert _location(_job("Europe"), zone="Europe/Paris") == 0.5
    assert _location(_job("New York"), zone="America/New_York") == 1.0


def test_pas_de_faux_positif_sur_sous_chaine():
    assert _location(_job("Parisot"), home="Paris") == 0.5


def test_aucun_lieu_personnel_en_dur():
    source = inspect.getsource(matching).lower()
    assert "abidjan" not in source and "ivoire" not in source


@pytest.mark.parametrize("func", [matching.build_features, matching.train_from_labels, pipeline.analyse])
def test_les_reglages_sont_acceptes_explicitement(func):
    params = inspect.signature(func).parameters
    assert "home_location" in params and "timezone" in params
    # obligatoires : un oubli lèverait une TypeError au lieu de retomber silencieusement en neutre
    assert params["home_location"].default is inspect.Parameter.empty
    assert params["timezone"].default is inspect.Parameter.empty
