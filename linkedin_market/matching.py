"""Score de matching profil <-> offre : régression logistique entraînée
par descente de gradient (NumPy, sans scikit-learn).

Chaque offre est décrite par quelques critères interprétables (couverture
des compétences, similarité textuelle, titre, séniorité, expérience,
télétravail, localisation, freelance). Tant que l'utilisateur n'a pas
annoté assez d'offres, des poids a priori sont utilisés ; dès qu'il y a
des annotations « pertinente / pas pertinente » des deux classes, le
modèle apprend ses propres poids et les sauvegarde.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import config, nlp
from .models import Item, Profile

FEATURES = [
    "couverture_competences", "similarite_texte", "similarite_titre", "seniorite",
    "experience", "teletravail", "localisation_compatible", "freelance",
]
# Poids a priori (sur des critères dans [0, 1]) : la couverture des
# compétences et la proximité du texte dominent.
PRIOR_WEIGHTS = np.array([2.5, 2.0, 1.2, 0.6, 0.5, 0.6, 0.8, 0.4])
PRIOR_BIAS = -4.3
MIN_LABELS = 6

_LOCATION_BLOCKERS = re.compile(
    r"us only|usa only|must be (?:based|located) in|resident of|eligible to work in the (?:us|uk|eu)"
    r"|sur site|on-?site|presentiel|hybride|hybrid|uniquement en france|work permit"
)
_LOCATION_OK = re.compile(r"worldwide|anywhere|monde|international|afrique|africa|cote d.ivoire|abidjan|emea|remote")
_FREELANCE = re.compile(r"freelance|independant|contract(?:or)?|contrat de mission|\bmission\b|consultant|portage|\btjm\b|daily rate|b2b")


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(nlp.tokens(a)), set(nlp.tokens(b))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


@dataclass
class JobFeatures:
    item: Item
    values: np.ndarray
    job_skills: list[str]
    missing_skills: list[str]


def build_features(jobs: list[Item], profile: Profile, keyword: str) -> list[JobFeatures]:
    if not jobs:
        return []
    tfidf = nlp.TfIdf([profile.text] + [f"{j.title} {j.text}" for j in jobs])
    vectors = tfidf.transform([profile.text] + [f"{j.title} {j.text}" for j in jobs])
    similarities = vectors[1:] @ vectors[0]
    profile_skills = set(profile.skills)
    profile_level = nlp.SENIORITY_LEVELS.get(profile.seniority)
    target_title = f"{keyword} {profile.title}"

    out = []
    for job, similarity in zip(jobs, similarities):
        full = f"{job.title} {job.text}"
        norm = nlp.normalize(f"{job.location} {full}")
        skills = nlp.extract_skills(full)
        coverage = len(profile_skills & set(skills)) / len(skills) if skills else 0.5

        job_level = nlp.SENIORITY_LEVELS.get(nlp.extract_seniority(job.title) or nlp.extract_seniority(job.text[:600]))
        seniority = 0.5 if job_level is None or profile_level is None else 1 - abs(job_level - profile_level) / 3

        required = nlp.extract_years(job.text)
        if required is None or profile.years_experience is None:
            experience = 0.5
        else:
            experience = 1.0 if profile.years_experience >= required else max(0.0, 1 - (required - profile.years_experience) / 5)

        if _LOCATION_BLOCKERS.search(norm):
            location = 0.0
        else:
            location = 1.0 if _LOCATION_OK.search(norm) else 0.5

        values = np.array([
            coverage,
            min(float(similarity) * 3, 1.0),   # cosinus TF-IDF typiquement < 0,35
            max(_token_jaccard(target_title, job.title), _token_jaccard(keyword, job.title)),
            seniority,
            experience,
            1.0 if nlp.is_remote(full) else 0.0,
            location,
            1.0 if _FREELANCE.search(norm) else 0.0,
        ])
        out.append(JobFeatures(job, values, skills, sorted(set(skills) - profile_skills)))
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class MatchingModel:
    weights: np.ndarray = field(default_factory=lambda: PRIOR_WEIGHTS.copy())
    bias: float = PRIOR_BIAS
    mean: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES)))
    std: np.ndarray = field(default_factory=lambda: np.ones(len(FEATURES)))
    trained: bool = False
    n_labels: int = 0
    train_accuracy: float | None = None
    loss_history: list[float] = field(default_factory=list)
    trained_at: str = ""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(((X - self.mean) / self.std) @ self.weights + self.bias)

    def importance(self) -> list[dict]:
        """Poids ramenés à l'échelle des critères bruts (effet d'un passage de 0 à 1)."""
        raw = self.weights / self.std
        return sorted(
            ({"feature": f, "weight": round(float(w), 3)} for f, w in zip(FEATURES, raw)),
            key=lambda r: abs(r["weight"]), reverse=True,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.3, epochs: int = 3000, l2: float = 0.05) -> None:
        """Descente de gradient sur l'entropie croisée + pénalité L2.

        Les critères sont standardisés pour que le pas d'apprentissage soit
        le même pour tous ; les classes déséquilibrées sont repondérées.
        """
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0))
        Xs = (X - self.mean) / self.std
        n = len(y)
        pos = max(y.sum(), 1)
        neg = max(n - y.sum(), 1)
        sample_w = np.where(y == 1, n / (2 * pos), n / (2 * neg))

        w = np.zeros(X.shape[1])
        b = 0.0
        self.loss_history = []
        for epoch in range(epochs):
            p = _sigmoid(Xs @ w + b)
            err = (p - y) * sample_w
            grad_w = Xs.T @ err / n + l2 * w
            grad_b = err.mean()
            w -= lr * grad_w
            b -= lr * grad_b
            if epoch % 25 == 0 or epoch == epochs - 1:
                eps = 1e-9
                loss = -np.mean(sample_w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) + l2 / 2 * w @ w
                self.loss_history.append(round(float(loss), 5))
        self.weights, self.bias = w, float(b)
        self.trained = True
        self.n_labels = n
        self.train_accuracy = round(float(((self.predict(X) >= 0.5) == y).mean()), 3)
        self.trained_at = datetime.now(timezone.utc).isoformat()

    # ---------------------------------------------------------- persistance
    def save(self) -> None:
        config.ensure_data_dirs()
        config.MODEL_FILE.write_text(json.dumps({
            "features": FEATURES, "weights": self.weights.tolist(), "bias": self.bias,
            "mean": self.mean.tolist(), "std": self.std.tolist(), "n_labels": self.n_labels,
            "train_accuracy": self.train_accuracy, "loss_history": self.loss_history,
            "trained_at": self.trained_at,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "MatchingModel":
        if not config.MODEL_FILE.exists():
            return cls()
        data = json.loads(config.MODEL_FILE.read_text(encoding="utf-8"))
        if data.get("features") != FEATURES:   # critères modifiés : modèle obsolète
            return cls()
        return cls(
            weights=np.array(data["weights"]), bias=data["bias"], mean=np.array(data["mean"]),
            std=np.array(data["std"]), trained=True, n_labels=data["n_labels"],
            train_accuracy=data.get("train_accuracy"), loss_history=data.get("loss_history", []),
            trained_at=data.get("trained_at", ""),
        )


def train_from_labels(profile: Profile, keyword: str, labels: dict[str, int], labeled_items: dict[str, Item]) -> MatchingModel | None:
    """Réentraîne sur toutes les annotations ; None si pas assez d'exemples."""
    items = [labeled_items[i] for i in labels if i in labeled_items]
    y = np.array([labels[i.id] for i in items], dtype=float)
    if len(items) < MIN_LABELS or y.min() == y.max():
        return None
    feats = build_features(items, profile, keyword)
    model = MatchingModel()
    model.fit(np.vstack([f.values for f in feats]), y)
    model.save()
    return model


def score_jobs(features: list[JobFeatures], model: MatchingModel) -> list[dict]:
    if not features:
        return []
    scores = model.predict(np.vstack([f.values for f in features]))
    rows = []
    for f, score in zip(features, scores):
        criteria = {name: round(float(v), 3) for name, v in zip(FEATURES, f.values)}
        rows.append({
            "id": f.item.id, "title": f.item.title, "company": f.item.company, "url": f.item.url,
            "location": f.item.location, "posted_at": f.item.posted_at,
            "score": round(float(score) * 100, 1),
            "criteria": criteria,
            "job_skills": f.job_skills, "missing_skills": f.missing_skills,
            "matched_skills": [s for s in f.job_skills if s not in f.missing_skills],
            # Fiche de lecture rapide de l'offre.
            "summary": nlp.key_sentences(f.item.text),
            "years_required": nlp.extract_years(f.item.text),
            "seniority_required": nlp.extract_seniority(f.item.title) or nlp.extract_seniority(f.item.text[:600]),
            "remote": bool(criteria["teletravail"]),
            "freelance": bool(criteria["freelance"]),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def skill_gap(scored: list[dict], profile: Profile) -> dict:
    """Compétences demandées par les offres, possédées ou manquantes.

    Le poids d'une offre dans le calcul est son score : un manque sur une
    offre qui te correspond par ailleurs compte plus.
    """
    demand: dict[str, float] = {}
    for row in scored:
        for skill in row["job_skills"]:
            demand[skill] = demand.get(skill, 0.0) + 0.5 + row["score"] / 200
    total = sum(1 for _ in scored) or 1
    owned = set(profile.skills)
    rows = sorted(
        ({"skill": s, "weight": round(w / total, 3), "owned": s in owned} for s, w in demand.items()),
        key=lambda r: r["weight"], reverse=True,
    )
    covered = sum(r["weight"] for r in rows if r["owned"])
    return {
        "skills": rows,
        "missing": [r for r in rows if not r["owned"]],
        "coverage": round(covered / (sum(r["weight"] for r in rows) or 1) * 100, 1),
    }


def uncertain_jobs(scored: list[dict], labels: dict[str, int], count: int) -> list[dict]:
    """Offres non annotées les plus proches de 50 % : celles qui apprennent le plus au modèle."""
    pending = [r for r in scored if r["id"] not in labels]
    pending.sort(key=lambda r: abs(r["score"] - 50))
    return pending[:count]
