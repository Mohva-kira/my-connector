"""Nettoyage du texte d'une offre LinkedIn : seul le contenu de l'offre est conservé."""

from linkedin_market import nlp

OFFRE = "Missions : développer des API Python. Profil : 3 ans d'expérience. Avantages : télétravail."


def test_fr_entete_et_pied_de_page_retires():
    brut = (
        "Candidature simplifiée Enregistrer Évaluation des candidatures en cours "
        f"À propos de l’offre d’emploi {OFFRE} Plus d’offres d’emploi Offre similaire LinkedIn Corporation © 2026"
    )
    assert nlp.strip_linkedin_chrome(brut) == OFFRE


def test_en_entete_et_pied_de_page_retires():
    brut = (
        "Easy Apply Save Show match details Meet the hiring team Jane Doe "
        f"About the job {OFFRE} Similar jobs Select language"
    )
    assert nlp.strip_linkedin_chrome(brut) == OFFRE


def test_sans_ancre_texte_conserve():
    assert nlp.strip_linkedin_chrome(OFFRE) == OFFRE


def test_idempotent():
    brut = f"Enregistrer À propos de l’offre d’emploi {OFFRE} Voir plus"
    once = nlp.strip_linkedin_chrome(brut)
    assert nlp.strip_linkedin_chrome(once) == once
