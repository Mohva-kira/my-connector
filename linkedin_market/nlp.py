"""Extraction statistique locale (sans LLM) : tokens, n-grammes, TF-IDF,
compétences, séniorité, années d'expérience, télétravail.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np

STOPWORDS = set("""
a à au aux avec ce ces cet cette dans de des du elle en et être eux il ils je la le les leur lui ma mais me même mes moi mon ne nos notre nous on ou où par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l m n s t y été étée étés est sont était sera ont avoir fait faire plus moins très tout tous toute toutes comme si aussi bien sans sous entre chez vers dont alors ainsi donc car afin cela ça ceci celui celle ceux leurs nos chaque autre autres peut peuvent doit doivent nous vous ici là déjà encore jour jours an ans année années mois semaine semaines h min
the of and to in for on with at by from as is are was were be been being an or not this that these those it its we our you your they their he she his her them us will would can could should may might must shall do does did have has had having but if then than so such into about over under more most other some any each all both no nor only own same too very just also via per etc
linkedin voir afficher postuler candidature candidater offre offres emploi job jobs poste postes entreprise company recrutement recruteur abonnés abonné followers suivre connexion relations message messages page pages profil profile commentaire commentaires like likes réagir partager share repost republié édité modifié publié promoted sponsorisé republication premium afficher plus see more show less moins apply applicants candidats candidat il y ago new nouveau nouvelle hashtag
""".split())

# Compétences canoniques -> motifs (regex) qui les détectent dans un texte normalisé.
SKILLS: dict[str, str] = {
    "python": r"python", "javascript": r"javascript|\bjs\b", "typescript": r"typescript",
    "java": r"\bjava\b(?!script)", "c#": r"c#|\.net|dotnet", "php": r"\bphp\b", "go": r"\bgolang\b|\bgo\b(?= lang| developer)",
    "rust": r"\brust\b", "kotlin": r"kotlin", "swift": r"\bswift\b", "sql": r"\bsql\b",
    "react": r"\breact(?:\.?js)?\b(?! native)", "react native": r"react native", "angular": r"angular",
    "vue": r"\bvue(?:\.?js)?\b", "next.js": r"next\.?js", "node.js": r"node\.?js|\bnode\b",
    "django": r"django", "fastapi": r"fastapi", "flask": r"\bflask\b", "laravel": r"laravel",
    "spring": r"spring ?boot|\bspring\b", "symfony": r"symfony", "flutter": r"flutter",
    "postgresql": r"postgres(?:ql)?", "mysql": r"mysql", "mongodb": r"mongo(?:db)?", "redis": r"\bredis\b",
    "elasticsearch": r"elastic ?search", "graphql": r"graphql", "rest api": r"\brest(?:ful)?\b|\bapi\b|\bapis\b",
    "microservices": r"micro-?services?", "aws": r"\baws\b|amazon web services", "azure": r"\bazure\b",
    "gcp": r"\bgcp\b|google cloud", "docker": r"docker", "kubernetes": r"kubernetes|\bk8s\b",
    "terraform": r"terraform", "ci/cd": r"ci ?/ ?cd|continuous (?:integration|delivery)|github actions|gitlab ci|jenkins",
    "devops": r"devops", "linux": r"\blinux\b", "git": r"\bgit\b", "cloud": r"\bcloud\b",
    "machine learning": r"machine learning|apprentissage automatique|\bml\b", "deep learning": r"deep learning",
    "ia générative": r"genai|gen ai|generative ai|ia generative|\bllms?\b|chatgpt|openai|claude|gpt",
    "nlp": r"\bnlp\b|traitement du langage", "data science": r"data scien", "data engineering": r"data engineer|etl|elt|pipeline de donnees|data pipeline",
    "power bi": r"power ?bi", "tableau": r"\btableau\b(?! de)", "excel": r"\bexcel\b", "pandas": r"pandas",
    "spark": r"\bspark\b|pyspark", "airflow": r"airflow", "dbt": r"\bdbt\b", "snowflake": r"snowflake",
    "rag": r"\brag\b|retrieval augmented", "langchain": r"langchain", "mlops": r"mlops",
    "cybersécurité": r"cyber ?securit|securite informatique|security", "sap": r"\bsap\b", "salesforce": r"salesforce",
    "odoo": r"\bodoo\b", "erp": r"\berp\b", "crm": r"\bcrm\b", "sirh": r"\bsirh\b|\bhris\b|\bats\b",
    "wordpress": r"wordpress", "shopify": r"shopify", "seo": r"\bseo\b", "ui/ux": r"\bui\b|\bux\b|figma|user experience",
    "mobile": r"\bmobile\b|android|\bios\b", "fintech": r"fintech|mobile money|paiement|payment",
    "blockchain": r"blockchain|web3|crypto", "agile": r"agile|agilite", "scrum": r"scrum",
    "gestion de projet": r"gestion de projets?|project manag|chef de projet|\bpmo\b", "product management": r"product (?:manag|owner)|\bpo\b",
    "transformation digitale": r"transformation (?:digitale|numerique)|digital transformation|digitalisation",
    "architecture logicielle": r"architect(?:ure|e) (?:logiciel|software|solution|technique)|solution architect|software architect",
    "tests automatisés": r"tests? (?:unitaires|automatis)|unit test|\btdd\b|pytest|jest|cypress|selenium",
    "anglais": r"\banglais\b|\benglish\b|bilingue|bilingual", "français": r"\bfrancais\b|\bfrench\b|francophone",
    "communication": r"communication", "leadership": r"leadership|management d.equipe|team lead",
    "consulting": r"consult(?:ing|ant)|conseil",
}
_SKILL_PATTERNS = {name: re.compile(p) for name, p in SKILLS.items()}

SENIORITY_LEVELS = {"junior": 0, "confirme": 1, "senior": 2, "lead": 3}
_SENIORITY_PATTERNS = [
    ("lead", r"\blead\b|principal|staff engineer|head of|\bcto\b|directeur|architect"),
    ("senior", r"senior|\bsr\b|expert|experimente"),
    ("junior", r"junior|\bjr\b|debutant|stagiaire|intern|alternan|entry level|graduate"),
    ("confirme", r"confirme|intermediaire|mid-level|\bmid\b"),
]
_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:ans|annees|years?|yrs)(?:\s+(?:d.|of)\s*(?:experience|exp))?")
_REMOTE = re.compile(r"full remote|100 ?% (?:remote|teletravail)|remote|teletravail|a distance|work from home|wfh")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#./-]*[a-z0-9+#]|[a-z]")


def normalize(text: str) -> str:
    """Minuscules, sans accents (« Télétravail » -> « teletravail »)."""
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c))


_STOP_NORMALIZED = {normalize(w) for w in STOPWORDS}


def _ui(*phrases: str) -> re.Pattern:
    """Motif insensible à la casse ; l'apostrophe typographique accepte aussi « ' »."""
    return re.compile("|".join(p.replace("’", "['’]") for p in phrases), re.IGNORECASE)


# Habillage de l'interface LinkedIn capturé avec la page d'une offre : encarts Premium,
# statut de candidature, encart recruteur.
_LINKEDIN_CHROME = _ui(
    r"Évaluation des candidatures en cours", r"Candidature simplifiée", r"\bEnregistrer\b",
    r"Utilisez l’IA pour évaluer si votre profil correspond",
    r"Obtenez des conseils générés par l’IA.{0,120}?avec Premium\.?",
    r"Afficher les détails de correspondance de votre profil",
    r"Découvrez comment vous vous positionnez par rapport .{0,80}?(?:candidats|Postuler)",
    r"Accédez à des informations exclusives sur les candidats.{0,200}?et plus encore\.?",
    r"Des recherches d’emploi plus rapides avec Premium.{0,600}?quel que soit le motif\.?",
    r"Activer une alerte emploi pour des postes similaires.{0,150}?(?:Désactivé|Activé)",
    r"Réactiver Premium", r"Promue par un recruteur", r"Réponses gérées en dehors de LinkedIn",
    r"Aucune info disponible pour le moment sur les réponses",
    r"Plus de \d+ (?:candidatures|personnes ont cliqué sur Postuler)",
    r"(?:…\s*plus\s+)?Voir plus\s*$",
    r"Personnes que vous pouvez contacter.{0,500}?Envoyer un message",
    r"Personnes que vous pouvez contacter", r"Rencontrez l’équipe de recrutement[^•]{0,80}•?",
    r"À propos de l’offre d’emploi",
)
# Début du bas de page (offres similaires, publicité Premium, pied de page, choix de langue) :
# tout ce qui suit est coupé.
_LINKEDIN_TAIL = _ui(
    r"(?:…\s*plus\s+)?Aimeriez-vous travailler avec nous", r"Plus d’offres d’emploi",
    r"Voir plus d’offres d’emploi similaires", r"Des recherches d’emploi plus intelligentes",
    r"Vous recherchez des talents \?", r"LinkedIn Corporation ©", r"Choisir une langue",
)


def strip_linkedin_chrome(text: str) -> str:
    """Retire de la page d'une offre l'interface LinkedIn (Premium, recruteurs, pied de page)."""
    tail = _LINKEDIN_TAIL.search(text or "")
    text = (text or "")[: tail.start()] if tail else (text or "")
    return re.sub(r"\s+", " ", _LINKEDIN_CHROME.sub(" ", text)).strip()


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text))
            if t not in _STOP_NORMALIZED and len(t) > 1 and not t.isdigit()]


def ngrams(toks: list[str], n_max: int = 3) -> list[str]:
    out = list(toks)
    for n in range(2, n_max + 1):
        out += [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return out


def extract_skills(text: str) -> list[str]:
    norm = normalize(text)
    return sorted(name for name, pattern in _SKILL_PATTERNS.items() if pattern.search(norm))


def extract_seniority(text: str) -> str:
    norm = normalize(text)
    for level, pattern in _SENIORITY_PATTERNS:
        if re.search(pattern, norm):
            return level
    return ""


def extract_years(text: str) -> int | None:
    values = [int(v) for v in _YEARS.findall(normalize(text)) if 0 < int(v) < 40]
    return max(values) if values else None


def is_remote(text: str) -> bool:
    return bool(_REMOTE.search(normalize(text)))


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s+[•·▪●◦–]\s+|\s+-\s+(?=[A-ZÀ-Ý])")
_KEY_HINTS = re.compile(
    r"mission|responsab|you will|you'll|vous (?:serez|allez|aurez)|votre role|your role|requirements|exigences"
    r"|profil recherche|looking for|recherch|must|required|requis|experience|build|develop|concevoir|lead|own"
)
_BOILERPLATE = re.compile(
    r"equal opportunit|benefits|avantages|about us|a propos|qui sommes|diversit|salary|salaire|we offer|nous offrons"
    r"|apply now|postulez|cookies|privacy|confidentialit|follow us|suivez"
)


def key_sentences(text: str, count: int = 3, max_len: int = 170) -> list[str]:
    """Les phrases les plus informatives d'une offre (mission, exigences, compétences),
    dans leur ordre d'origine et tronquées pour une lecture rapide."""
    sentences = [s.strip(" -•·") for s in _SENTENCE_SPLIT.split(text or "")]
    scored = []
    for pos, sentence in enumerate(sentences):
        if not 30 <= len(sentence) <= 400:
            continue
        norm = normalize(sentence)
        if _BOILERPLATE.search(norm):
            continue
        # Une simple énumération de compétences n'apporte rien de plus que la ligne ✓/✗.
        if len([t for t in tokens(sentence) if not extract_skills(t)]) < 4:
            continue
        score = 2 * len(extract_skills(sentence)) + len(_KEY_HINTS.findall(norm))
        if score:
            scored.append((score, pos, sentence))
    best = sorted(sorted(scored, reverse=True)[:count], key=lambda t: t[1])
    return [s if len(s) <= max_len else s[: max_len - 1].rsplit(" ", 1)[0] + "…" for _, _, s in best]


class TfIdf:
    """TF-IDF minimal en NumPy (unigrammes + bigrammes), vecteurs normalisés L2."""

    def __init__(self, documents: list[str], max_features: int = 3000, min_df: int = 1):
        grams = [ngrams(tokens(d), 2) for d in documents]
        df = Counter(g for doc in grams for g in set(doc))
        vocab = [g for g, c in df.most_common() if c >= min_df][:max_features]
        self.index = {g: i for i, g in enumerate(vocab)}
        n_docs = max(len(documents), 1)
        self.idf = np.array([math.log((1 + n_docs) / (1 + df[g])) + 1 for g in vocab])

    def transform(self, documents: list[str]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(self.index)))
        for row, doc in enumerate(documents):
            for gram, count in Counter(ngrams(tokens(doc), 2)).items():
                col = self.index.get(gram)
                if col is not None:
                    matrix[row, col] = 1 + math.log(count)
        matrix *= self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)
