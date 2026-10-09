"""Historique des éléments collectés et annotations en base (vague 1, sous-lot 1b)."""

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from linkedin_market import storage, trends
from linkedin_market.db import models, session, users
from linkedin_market.db.repositories import items as items_repo
from linkedin_market.models import Item

pytestmark = pytest.mark.db


def job(n: str, title: str = "Data Engineer", keyword: str = "Data Engineer", **extra) -> Item:
    return Item(
        kind="job",
        id=f"job:{n}",
        url=f"https://www.linkedin.com/jobs/view/{n}/",
        title=title,
        text="Python et Spark, full remote.",
        keyword=keyword,
        company="Acme",
        **{"posted_at": date.today().isoformat(), **extra},
    )


def rows():
    with session.transaction() as s:
        return s.execute(select(models.CollectedItem).order_by(models.CollectedItem.external_id)).scalars().all()


def test_item_from_dict_accepte_l_ancienne_cle_source():
    item = Item.from_dict({"source": "post", "id": "post:1", "url": "u", "title": "t", "text": "x", "keyword": "k"})
    assert item.kind == "post" and item.source_code == "linkedin"


def test_dedoublonnage_de_l_historique():
    storage.append_items([job("1"), job("2")])
    storage.append_items([job("1", title="Senior Data Engineer"), job("1", title="Lead Data Engineer")])
    found = rows()
    assert [r.external_id for r in found] == ["1", "2"]
    assert found[0].title == "Lead Data Engineer"  # dernière version gagnante
    assert found[0].kind == "job" and found[0].source_code == "linkedin" and "python" in found[0].skills
    assert {i.id for i in storage.load_items("data engineer")} == {"job:1", "job:2"}


def test_offre_recollectee_sous_un_second_mot_cle():
    """Une offre recollectée sous « Python » reste aussi dans les tendances de « Data Engineer »."""
    storage.append_items([job("1", keyword="Data Engineer")])
    storage.append_items([job("1", keyword=" python "), job("2", keyword="Python")])
    storage.append_items([job("1", keyword="DATA ENGINEER")])  # déjà connu : pas de doublon
    assert rows()[0].keywords == ["data engineer", "python"]
    assert {i.id for i in storage.load_items("Data Engineer")} == {"job:1"}
    python = storage.load_items("Python")
    assert {i.id for i in python} == {"job:1", "job:2"}
    assert {i.keyword for i in python} == {"Python"}  # mot-clé demandé
    assert storage.find_item("job:1").keyword == "data engineer"


def test_set_label_element_absent():
    storage.append_items([job("1")])
    assert storage.set_label("job:1", 1) == 1 and storage.load_labels() == {"job:1": 1}
    assert rows()[0].purge_after is None
    assert storage.set_label("job:404", 1) is None and storage.load_labels() == {"job:1": 1}
    assert storage.set_label("job:1", None) == 0 and storage.load_labels() == {}


def test_ingestion_par_lots():
    storage.append_items([job(str(n)) for n in range(250)])
    assert len(rows()) == 250


def test_person_sans_texte_et_competences_extraites():
    person = Item(
        kind="person",
        id="person:jane-doe",
        url="https://www.linkedin.com/in/jane-doe/",
        title="Data Engineer Python",
        text="Jane Doe\nData Engineer Python AWS\nParis",
        keyword="Data Engineer",
    )
    storage.append_items([person])
    row = rows()[0]
    assert row.text is None and {"python", "aws"} <= set(row.skills)
    loaded = storage.find_item("person:jane-doe")
    assert "Jane" not in loaded.text and "python" in loaded.text
    assert any(r["skill"] == "python" for r in trends.skill_demand(storage.load_items("Data Engineer")))


def test_purge_after_a_l_insertion():
    post = Item(kind="post", id="post:9", url="u", title="t", text="Python", keyword="k")
    storage.append_items([job("1", keyword="k"), post])
    by_kind = {r.kind: r for r in rows()}
    assert by_kind["job"].purge_after - by_kind["job"].collected_at == timedelta(days=90)
    assert by_kind["post"].purge_after - by_kind["post"].collected_at == timedelta(days=30)


def test_fenetre_de_30_jours_en_sql():
    old = (date.today() - timedelta(days=45)).isoformat()
    storage.append_items([job("1"), job("2", posted_at=old), job("3", keyword="Autre")])
    storage.append_items([Item(kind="job", id="job:4", url="u", title="t", text="x", keyword="Data Engineer")])
    assert {i.id for i in storage.load_items("DATA ENGINEER")} == {"job:1", "job:4"}  # sans date : collecte
    assert {i.id for i in storage.load_items("Data Engineer", window_days=3650)} == {"job:1", "job:2", "job:4"}
    assert storage.load_items("Data%") == []  # joker échappé


def test_annotations_et_purge_des_offres_annotees():
    storage.append_items([job("1"), job("2")])
    storage.save_labels({"job:1": 1, "job:2": 0, "job:999": 1})  # orpheline ignorée
    assert storage.load_labels() == {"job:1": 1, "job:2": 0}
    assert set(storage.load_labeled_items()) == {"job:1", "job:2"}
    found = {r.external_id: r for r in rows()}
    assert found["1"].purge_after is None and found["2"].purge_after is None
    storage.append_items([job("1", title="Nouveau titre")])  # une recollecte ne relance pas la purge
    assert {r.external_id: r for r in rows()}["1"].purge_after is None
    storage.save_labels({"job:2": 1})
    assert storage.load_labels() == {"job:2": 1}


def test_find_item():
    storage.append_items([job("7")])
    assert storage.find_item("job:7").title == "Data Engineer"
    assert storage.find_item("job:8") is None and storage.find_item("n'importe quoi") is None


def test_filtre_user_id(db_user_id):
    """I-01 : chaque requête des dépôts filtre sur user_id."""
    storage.append_items([job("1")])
    storage.save_labels({"job:1": 1})
    with session.transaction() as s:
        other = users.ensure_user(s, "autre@example.org")
        assert items_repo.window_items(s, other, "Data Engineer", 30) == []
        assert items_repo.find_item(s, other, "job:1") is None
        assert items_repo.get_labels(s, other) == {} and items_repo.labeled_items(s, other) == []
        assert len(items_repo.window_items(s, db_user_id, "Data Engineer", 30)) == 1
