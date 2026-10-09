"""Structures de données partagées par tout le pipeline (sérialisables JSON)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SearchRequest:
    keyword: str
    location: str = ""  # vide = monde entier
    remote_only: bool = True
    caution: str = "normal"
    max_items: int = 30
    profile_source: str = "auto"  # auto | file | linkedin
    extra_skills: list[str] = field(default_factory=list)
    years_experience: int | None = None
    seniority: str = ""  # junior | confirme | senior | lead


@dataclass
class Item:
    """Un élément collecté : offre, post ou profil (``kind``), sur une plateforme (``source_code``)."""

    kind: str  # job | post | person
    id: str
    url: str
    title: str
    text: str
    keyword: str
    company: str = ""
    location: str = ""
    posted_at: str = ""  # date ISO (approximée depuis « il y a 3 j »)
    scraped_at: str = ""
    source_code: str = "linkedin"  # plateforme d'origine (sources.code)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Item:
        """Accepte l'ancienne clé ``source`` (job | post | person) des historiques JSON."""
        data = dict(data)
        if "kind" not in data and data.get("source") in ("job", "post", "person"):
            data["kind"] = data.pop("source")
        known = {k: data.get(k, "") for k in cls.__dataclass_fields__}
        known["source_code"] = data.get("source_code") or "linkedin"
        return cls(**known)


@dataclass
class Profile:
    text: str
    origin: str  # chemin du fichier ou « linkedin »
    title: str = ""
    skills: list[str] = field(default_factory=list)
    years_experience: int | None = None
    seniority: str = ""
