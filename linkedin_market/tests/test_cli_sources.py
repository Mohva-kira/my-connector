"""CLI ``--sources`` : mêmes préconditions que l'API (I-16, I-06, P-3), refus clairs et code de sortie non nul."""

import json
import logging
from pathlib import Path

import httpx
import pytest

from linkedin_market import __main__ as cli
from linkedin_market import activity, config, consent, pipeline, storage
from linkedin_market.sources import base
from linkedin_market.tests.test_runs_sources import ft_offer, with_drivers, with_france_travail
from linkedin_market.tests.test_source_browser import FakeDriver, html, wttj_driver
from linkedin_market.tests.test_source_france_travail import ok

pytestmark = pytest.mark.db

BASE = ["-k", "Data Engineer", "--caution", "prudent", "--interpreter", "local", "--no-interactive", "--label", "0"]


@pytest.fixture(autouse=True)
def cli_env(monkeypatch, tmp_path):
    """Base déjà migrée (pas de bootstrap), pauses nulles, profil disponible, aucun navigateur réel."""
    monkeypatch.setattr(cli, "prepare_database", lambda: None)
    monkeypatch.setattr(base, "human_pause", lambda *a, **k: None)
    root = tmp_path / "project"
    root.mkdir()
    (root / "profil.json").write_text(
        json.dumps({"title": "Data Engineer", "skills": ["python", "spark"], "text": "Data Engineer Python"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", root)

    def no_browser():
        raise AssertionError("aucun navigateur ne doit démarrer sur un refus")

    monkeypatch.setattr("linkedin_market.browser.create_driver", no_browser)


def refused(*argv: str) -> str:
    with pytest.raises(SystemExit) as caught:
        cli.main([*BASE, *argv])
    assert caught.value.code not in (0, None)  # message en chaîne : code de sortie 1
    return str(caught.value.code)


def test_sources_par_defaut_linkedin_et_analyse_des_codes():
    assert cli.parse_args(["-k", "x"]).sources == ["linkedin"]
    assert cli.parse_args(["--sources", "LinkedIn, malt,malt,france_travail"]).sources == [
        "linkedin",
        "malt",
        "france_travail",
    ]
    with pytest.raises(SystemExit):
        cli.parse_args(["--sources", " , "])


def test_aide_documente_sources(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(["--help"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "--sources" in out and "Paramètres" in out and "FRANCE_TRAVAIL_CLIENT_ID" in out


@pytest.mark.parametrize("code", ["linkedin", "malt", "wttj"])
def test_source_navigateur_sans_consentement_renvoie_vers_l_interface_web(code):
    message = refused("--sources", code)
    assert "Accepte d'abord les risques" in message and "Paramètres → Sources et consentements" in message
    assert activity.block_status(source=code) is None


def test_un_seul_consentement_manquant_refuse_toute_la_demande(accept_risks):
    accept_risks("malt")
    assert "Welcome to the Jungle" in refused("--sources", "malt,wttj")


def test_indeed_refuse_import_manuel_via_web_uniquement():
    assert "interface web" in refused("--sources", "indeed")


def test_source_inconnue_refusee():
    assert "inconnue" in refused("--sources", "monster")


def test_france_travail_exige_ses_identifiants():
    assert "FRANCE_TRAVAIL_CLIENT_ID" in refused("--sources", "france_travail")


def test_repos_de_6_h_respecte(accept_risks):
    accept_risks("malt")
    activity.record("blocked", where="collecte", source="malt")
    assert "Malt a interrompu la navigation" in refused("--sources", "malt")


def test_repos_linkedin_respecte_pour_le_defaut(accept_risks):
    accept_risks("linkedin")
    activity.record("blocked", where="collecte", source="linkedin")
    assert "LinkedIn a interrompu la navigation" in refused()


def test_collecte_multi_sources_puis_analyse(accept_risks, monkeypatch, capsys):
    accept_risks("wttj")
    with_france_travail(monkeypatch, [ok([ft_offer()])])
    with_drivers(monkeypatch, wttj_driver())
    cli.main([*BASE, "--no-open", "--sources", "france_travail,wttj"])
    assert "Rapport :" in capsys.readouterr().out
    assert sorted({i.source_code for i in storage.load_items("Data Engineer") if i.kind == "job"}) == [
        "france_travail",
        "wttj",
    ]


def test_source_bloquee_pendant_la_collecte_rapport_partiel_et_repos(accept_risks, monkeypatch, capsys):
    accept_risks("malt")
    accept_risks("wttj")
    challenge = FakeDriver({"": html("blocked_challenge.html")})
    with_drivers(monkeypatch, challenge, wttj_driver())
    cli.main([*BASE, "--no-open", "--sources", "malt,wttj"])
    assert "Rapport :" in capsys.readouterr().out
    assert activity.block_status(source="malt") is not None and activity.block_status(source="wttj") is None
    assert len(challenge.gets) == 1  # aucune page après le défi (I-06)


def test_france_travail_en_echec_ignoree_sans_arret(monkeypatch, capsys):
    with_france_travail(monkeypatch, [httpx.Response(503)] * 3)
    cli.main([*BASE, "--no-open", "--sources", "france_travail"])
    assert "Rapport :" in capsys.readouterr().out


def test_no_scrape_ignore_les_sources_et_leurs_controles(monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "collect", lambda *a, **k: pytest.fail("aucune collecte avec --no-scrape"))
    cli.main([*BASE, "--no-open", "--no-scrape", "--sources", "malt"])
    assert "Rapport :" in capsys.readouterr().out


def test_le_cli_transmet_lieu_et_fuseau_a_l_analyse(monkeypatch):
    from linkedin_market import settings

    settings.save_settings({"home_location": "Lyon", "timezone": "Europe/Paris"})
    seen: list[tuple[str, str]] = []
    real = cli.analyse

    def spy(request, prof, home_location="", timezone=""):
        seen.append((home_location, timezone))
        return real(request, prof, home_location, timezone)

    monkeypatch.setattr(cli, "analyse", spy)
    cli.main([*BASE, "--no-open", "--no-scrape"])
    assert seen == [("Lyon", "Europe/Paris")]


def test_blocage_en_cours_de_collecte_code_0_rapport_partiel_sans_double_journal(
    accept_risks, monkeypatch, capsys, caplog
):
    """Décision conservatrice : un blocage survenu PENDANT la collecte ne change pas le code de sortie (0)."""
    accept_risks("wttj")

    def challenge_at_second_offer(driver, url):
        if len(driver.gets) == 3:
            driver.page_source = html("blocked_challenge.html")

    driver = wttj_driver(on_get=challenge_at_second_offer)
    with_drivers(monkeypatch, driver)
    caplog.set_level(logging.INFO)
    cli.main([*BASE, "--no-open", "--sources", "wttj"])  # aucun SystemExit : code de sortie 0
    out = capsys.readouterr().out
    assert "Collecte interrompue par Welcome to the Jungle : rapport établi sur les données déjà collectées" in out
    assert "Rapport :" in out
    assert [i.id for i in storage.load_items("Data Engineer") if i.source_code == "wttj"] == [
        "job:wttj-data-engineer_paris"
    ]  # offre prise avant le défi, conservée dans le rapport partiel
    assert activity.block_status(source="wttj") is not None  # repos de 6 h
    assert len(driver.gets) == 3 and driver.quit_called == 1  # rien après le défi (I-06), navigateur fermé
    assert Path(out.split("Rapport : ")[1].splitlines()[0]).exists()  # rapport partiel écrit
    # un seul journal par blocage : le récapitulatif CLI est imprimé, pas journalisé de nouveau
    assert not [r for r in caplog.records if "Collecte interrompue par" in r.getMessage()]
    assert len([r for r in caplog.records if r.name == "linkedin_market" and "repos de 6 h" in r.getMessage()]) == 1


def test_verification_manuelle_malt_wttj_est_un_blocage_sans_attente(accept_risks, monkeypatch, capsys):
    """Sans ``on_manual_check`` (CLI), un défi Malt / WTTJ arrête la source : ni attente, ni contournement."""
    accept_risks("malt")

    def no_input(*a, **k):
        raise AssertionError("aucune saisie ni attente pour une vérification manuelle Malt / WTTJ")

    monkeypatch.setattr("builtins.input", no_input)
    challenge = FakeDriver({"": html("blocked_challenge.html")})
    with_drivers(monkeypatch, challenge)
    cli.main([*BASE, "--no-open", "--sources", "malt"])
    out = capsys.readouterr().out
    assert "Collecte interrompue par Malt : rapport établi sur les données déjà collectées" in out
    assert "6 h" in out
    assert activity.block_status(source="malt") is not None
    assert len(challenge.gets) == 1 and challenge.quit_called == 1


def test_profil_absent_sans_linkedin_message_actionnable(monkeypatch):
    with_france_travail(monkeypatch, [ok([ft_offer()])])
    (config.PROJECT_ROOT / "profil.json").unlink()
    message = refused("--sources", "france_travail")
    assert "profil.md" in message and "cv.pdf" in message
    assert "--sources linkedin" in message and "interface web" in message


def test_plafond_i07_applique_sur_le_chemin_cli_multi_sources(accept_risks, monkeypatch, capsys):
    """``--max-items`` au-delà de la prudence : chaque source reçoit le plafond (100 en « prudent »)."""
    accept_risks("wttj")
    seen: dict[str, int] = {}

    def fake_ft(request):
        seen["france_travail"] = request.max_items
        return pipeline.SourceReport("france_travail", 0)

    def fake_browser(code, request, check):
        seen[code] = request.max_items
        return pipeline.SourceReport(code, 0)

    monkeypatch.setattr(pipeline, "_collect_france_travail", fake_ft)
    monkeypatch.setattr(pipeline, "_collect_browser", fake_browser)
    monkeypatch.setattr(pipeline.config, "france_travail_configured", lambda: True)
    cli.main([*BASE, "--no-open", "--max-items", "500", "--sources", "france_travail,wttj"])
    assert seen == {"france_travail": 100, "wttj": 100}
    assert config.CAUTION_LEVELS["prudent"].max_items == 15
    assert "Rapport :" in capsys.readouterr().out


def test_consentement_revoque_entre_verification_et_collecte(accept_risks, monkeypatch):
    accept_risks("wttj")
    real = cli.check_cli_sources

    def check_then_revoke(codes):
        real(codes)  # la vérification initiale passe
        consent.revoke("wttj")  # puis l'utilisateur révoque avant la collecte

    monkeypatch.setattr(cli, "check_cli_sources", check_then_revoke)
    message = refused("--sources", "wttj")  # code de sortie non nul, aucun navigateur démarré
    assert "Accepte d'abord les risques" in message and "Paramètres → Sources et consentements" in message


def _verification_a_la_connexion(monkeypatch, driver: FakeDriver) -> None:
    """LinkedIn ne valide pas la vérification : ``session.login`` pose ``blocked`` et lève l'exception réelle."""
    from linkedin_market import session

    with_drivers(monkeypatch, driver)
    monkeypatch.setattr(session, "login", lambda drv, **kw: session._blocked_at_login())


def test_verification_linkedin_a_la_connexion_sortie_propre_code_1(accept_risks, monkeypatch, capsys):
    """Refus avant toute collecte LinkedIn : message VERIFICATION_NOT_DONE, code 1, sans trace, navigateur fermé."""
    from linkedin_market import session

    accept_risks("linkedin")
    driver = FakeDriver({"": "<html></html>"})
    _verification_a_la_connexion(monkeypatch, driver)
    with pytest.raises(SystemExit) as caught:
        cli.main([*BASE, "--no-open"])
    assert caught.value.code == session.VERIFICATION_NOT_DONE
    assert driver.quit_called == 1 and driver.gets == []  # navigateur fermé, aucune navigation (I-06)
    assert activity.block_status(source="linkedin")["where"] == "connexion"  # repos de 6 h posé
    assert "Rapport :" not in capsys.readouterr().out


def test_verification_linkedin_a_la_connexion_en_multi_sources_code_1(accept_risks, monkeypatch, capsys):
    """LinkedIn en tête : la connexion échoue avant toute collecte, les sources suivantes ne démarrent pas."""
    from linkedin_market import session

    accept_risks("linkedin")
    accept_risks("wttj")
    driver = FakeDriver({"": "<html></html>"})
    _verification_a_la_connexion(monkeypatch, driver)  # un seul navigateur : un second ferait échouer le test
    with pytest.raises(SystemExit) as caught:
        cli.main([*BASE, "--no-open", "--sources", "linkedin,wttj"])
    assert caught.value.code == session.VERIFICATION_NOT_DONE
    assert driver.quit_called == 1
    assert activity.block_status(source="wttj") is None  # WTTJ jamais touchée
    assert "Rapport :" not in capsys.readouterr().out
