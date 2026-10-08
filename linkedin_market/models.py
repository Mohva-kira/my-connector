"""Structures de données partagées par tout le pipeline (sérialisables JSON)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SearchRequest:
    keyword: str
    location: str = ""                 # vide = monde entier
    remote_only: bool = True
    caution: str = "normal"
    max_items: int = 30
    profile_source: str = "auto"       # auto | file | linkedin
    extra_skills: list[str] = field(default_factory=list)
    years_experience: int | None = None
    seniority: str = ""                # junior | confirme | senior | lead


@dataclass
class Item:
    """Un élément récupéré sur LinkedIn : offre, post ou profil."""

    source: str                        # job | post | person
    id: str
    url: str
    title: str
    text: str
    keyword: str
    company: str = ""
    location: str = ""
    posted_at: str = ""                # date ISO (approximée depuis « il y a 3 j »)
    scraped_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Item":
        known = {k: data.get(k, "") for k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Profile:
    text: str
    origin: str                        # chemin du fichier ou « linkedin »
    title: str = ""
    skills: list[str] = field(default_factory=list)
    years_experience: int | None = None
    seniority: str = ""
