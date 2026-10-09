"""Collecte LinkedIn : offres d'emploi, posts, profils et profil de l'utilisateur.

Les sélecteurs CSS de LinkedIn changent souvent. Chaque extraction essaie
donc une liste de sélecteurs connus puis se rabat sur le texte brut de
``<main>`` : on perd un peu de précision mais le pipeline continue.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import quote_plus

from selenium.common.exceptions import WebDriverException

from . import config, nlp
from .browser import human_pause
from .models import Item, SearchRequest

log = logging.getLogger(__name__)

JOBS_PER_PAGE = 25
PEOPLE_PER_PAGE = 10
MAX_POST_SCROLLS = 12

_UNIT_DAYS = [
    (r"min(?:ute)?s?", 0),
    (r"h(?:eures?|ours?|rs?)?", 0),
    (r"j(?:ours?)?|d(?:ays?)?", 1),
    (r"sem(?:aines?|\.)?|w(?:eeks?)?", 7),
    (r"mois|mo(?:nths?)?", 30),
    (r"ans?|y(?:ears?|rs?)?", 365),
]
_RELATIVE_LONG = re.compile(
    r"(?:il y a\s+)?(\d+)\s*"
    r"(minutes?|heures?|hours?|jours?|days?|semaines?|weeks?|mois|months?|ans?|years?)(?:\s+ago)?",
    re.IGNORECASE,
)
_RELATIVE_SHORT = re.compile(r"^\s*(\d+)\s*(min|h|j|d|sem\.?|w|mo|mois|an|y|yr)\b", re.IGNORECASE)


def parse_relative_date(text: str, now: datetime | None = None) -> str:
    """« il y a 2 semaines », « 3 j », « 1mo ago »… -> date ISO approximative."""
    now = now or datetime.now(UTC)
    match = _RELATIVE_LONG.search(text or "") or _RELATIVE_SHORT.search(text or "")
    if not match:
        return ""
    value, unit = int(match.group(1)), match.group(2).lower()
    for pattern, days in _UNIT_DAYS:
        if re.fullmatch(pattern, unit):
            return (now - timedelta(days=value * days)).date().isoformat()
    return ""


# JavaScript : texte du premier sélecteur présent, sinon de <main>.
_JS_TEXT = """
const sels = arguments[0];
for (const s of sels) {
  const el = document.querySelector(s);
  if (el && el.textContent.trim().length > 0) return el.textContent;
}
const main = document.querySelector('main');
return main ? main.innerText : document.body.innerText;
"""

_JS_SCROLL_LISTS = """
window.scrollTo(0, document.body.scrollHeight);
for (const el of document.querySelectorAll('main *')) {
  if (el.scrollHeight > el.clientHeight + 50 && el.clientHeight > 200) el.scrollTop = el.scrollHeight;
}
"""

_JS_POSTS = """
const out = [];
const nodes = document.querySelectorAll('[data-urn^="urn:li:activity"], div.feed-shared-update-v2');
for (const n of nodes) {
  const pick = (sels) => { for (const s of sels) { const e = n.querySelector(s); if (e) return e.textContent.trim(); } return ''; };
  out.push({
    urn: n.getAttribute('data-urn') || '',
    text: pick(['.update-components-text', '.feed-shared-inline-show-more-text', '.feed-shared-text']) || n.innerText,
    author: pick(['.update-components-actor__title', '.update-components-actor__name']),
    headline: pick(['.update-components-actor__description']),
    when: pick(['.update-components-actor__sub-description']),
  });
}
return out;
"""  # noqa: E501

_JS_PEOPLE = """
const out = {};
for (const a of document.querySelectorAll('main a[href*="/in/"]')) {
  const m = a.href.match(/linkedin\\.com\\/in\\/([^/?#]+)/);
  if (!m || out[m[1]]) continue;
  const card = a.closest('li') || a.closest('div[data-chameleon-result-urn]') || a.parentElement;
  out[m[1]] = card ? card.innerText : a.innerText;
}
return out;
"""


_HIRING = re.compile(
    r"^(?P<company>.+?)\s+(?:hiring|recrute|recherche)\s+(?P<title>.+?)(?:\s+(?:in|à|a)\s+.+)?$", re.IGNORECASE
)


def parse_job_page_title(page_title: str) -> tuple[str, str]:
    """« Titre | Entreprise | LinkedIn » ou « Entreprise hiring Titre in Lieu | LinkedIn » -> (titre, entreprise).

    Le titre de l'onglet est bien plus stable que les sélecteurs CSS de la page.
    """
    parts = [p.strip() for p in re.split(r"\s+[|\u2013-]\s+", page_title or "") if p.strip()]
    parts = [p for p in parts if p.lower() != "linkedin"]
    if len(parts) >= 2:
        return parts[0][:150], parts[1][:100]
    if len(parts) == 1:
        match = _HIRING.match(parts[0])
        if match:
            return match["title"].strip()[:150], match["company"].strip()[:100]
        return parts[0][:150], ""
    return "", ""


def _short(text: str, limit: int) -> str:
    """Première ligne non vide, ou chaîne vide si elle dépasse ``limit`` (texte de page brut)."""
    first = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
    return first if len(first) <= limit else ""


def _text(driver, selectors: list[str]) -> str:
    try:
        return (driver.execute_script(_JS_TEXT, selectors) or "").strip()
    except WebDriverException:
        return ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _scroll(driver, caution: config.Caution, times: int = 3) -> None:
    for _ in range(times):
        driver.execute_script(_JS_SCROLL_LISTS)
        human_pause(caution, 0.3)


class LinkedInScraper:
    """Collecte page par page.

    ``check`` est appelé avant chaque chargement de page : il lève une exception si l'utilisateur a annulé (E-22).
    Après un blocage (``blocked``), plus aucune page n'est chargée (I-06). ``partial`` contient les éléments déjà
    construits par la méthode de collecte en cours (conservés si elle est interrompue).
    """

    def __init__(self, driver, request: SearchRequest, *, check: Callable[[], None] | None = None):
        self.driver = driver
        self.request = request
        self.caution = config.CAUTION_LEVELS.get(request.caution, config.CAUTION_LEVELS[config.DEFAULT_CAUTION])
        self.now = datetime.now(UTC)
        self._check = check or (lambda: None)
        self.blocked = False
        self.partial: list[Item] = []

    def _blocked(self) -> bool:
        url = self.driver.current_url
        if "/checkpoint" in url or "/authwall" in url or "/login" in url:
            self.blocked = True
            log.error(
                "LinkedIn a interrompu la navigation (%s). Arrêt de la collecte : "
                "relance plus tard avec le niveau « prudent ».",
                url,
            )
            return True
        return False

    def _get(self, url: str) -> bool:
        if self.blocked:
            return False  # arrêt immédiat après un blocage (I-06)
        self._check()
        try:
            self.driver.get(url)
        except WebDriverException as exc:
            log.warning("Chargement impossible %s : %s", url, exc.msg)
            return False
        human_pause(self.caution)
        return not self._blocked()

    # ----------------------------------------------------------- offres
    def jobs(self) -> list[Item]:
        self.partial = []
        keyword = quote_plus(self.request.keyword)
        params = f"keywords={keyword}&f_TPR=r{config.TREND_WINDOW_DAYS * 86400}"
        if self.request.remote_only:
            params += "&f_WT=2"
        if self.request.location:
            params += f"&location={quote_plus(self.request.location)}"

        job_ids: list[str] = []
        start = 0
        while len(job_ids) < self.request.max_items:
            if not self._get(f"https://www.linkedin.com/jobs/search/?{params}&start={start}"):
                break
            _scroll(self.driver, self.caution)
            found = re.findall(r"(?:/jobs/view/|data-job-id=\"|currentJobId=)(\d{6,})", self.driver.page_source)
            new = [j for j in dict.fromkeys(found) if j not in job_ids]
            if not new:
                break
            job_ids.extend(new)
            start += JOBS_PER_PAGE
        job_ids = job_ids[: self.request.max_items]
        log.info("%d offres trouvées, lecture du détail...", len(job_ids))

        items = self.partial = []
        for job_id in job_ids:
            url = f"https://www.linkedin.com/jobs/view/{job_id}/"
            if not self._get(url):
                break
            h1 = _text(self.driver, ["h1"])
            description = _text(
                self.driver,
                [
                    ".jobs-description__content",
                    "#job-details",
                    ".jobs-box__html-content",
                    ".jobs-description",
                    "article",
                ],
            )
            top_card = _text(
                self.driver,
                [
                    ".job-details-jobs-unified-top-card__primary-description-container",
                    ".job-details-jobs-unified-top-card__tertiary-description-container",
                    ".jobs-unified-top-card",
                ],
            )
            company = _text(
                self.driver,
                [
                    ".job-details-jobs-unified-top-card__company-name",
                    ".jobs-unified-top-card__company-name",
                ],
            )
            title, page_company = parse_job_page_title(self.driver.title)
            title = title or _short(h1, 150)
            company = page_company or _short(company, 100)
            items.append(
                Item(
                    kind="job",
                    id=f"job:{job_id}",
                    url=url,
                    title=_clean(title),
                    text=nlp.strip_linkedin_chrome(description),
                    keyword=self.request.keyword,
                    company=_clean(company),
                    location=_clean(top_card.split("·")[0])[:120],
                    posted_at=parse_relative_date(top_card[:400], self.now) or self.now.date().isoformat(),
                    scraped_at=self.now.isoformat(),
                )
            )
        return items

    # ------------------------------------------------------------ posts
    def posts(self) -> list[Item]:
        self.partial = []
        keyword = quote_plus(self.request.keyword)
        url = (
            f"https://www.linkedin.com/search/results/content/?keywords={keyword}"
            "&datePosted=%22past-month%22&sortBy=%22date_posted%22"
        )
        if not self._get(url):
            return []
        raw: dict[str, dict] = {}
        for _ in range(MAX_POST_SCROLLS):
            for post in self.driver.execute_script(_JS_POSTS) or []:
                text = _clean(post.get("text", ""))
                key = post.get("urn") or text[:80]
                if len(text) > 40 and key not in raw:
                    raw[key] = post | {"text": text}
            if len(raw) >= self.request.max_items:
                break
            _scroll(self.driver, self.caution, times=1)
        if not raw:
            log.warning("Aucun post détecté : la structure de la page a peut-être changé.")

        items = []
        for key, post in list(raw.items())[: self.request.max_items]:
            urn = post.get("urn", "")
            items.append(
                Item(
                    kind="post",
                    id=f"post:{urn or abs(hash(key))}",
                    url=f"https://www.linkedin.com/feed/update/{urn}/" if urn else url,
                    title=_clean(post.get("headline") or post.get("author", "")),
                    text=post["text"],
                    keyword=self.request.keyword,
                    company=_clean(post.get("author", "")),
                    posted_at=parse_relative_date(post.get("when", ""), self.now) or self.now.date().isoformat(),
                    scraped_at=self.now.isoformat(),
                )
            )
        return items

    # ---------------------------------------------------------- profils
    def people(self) -> list[Item]:
        self.partial = []
        keyword = quote_plus(self.request.keyword)
        cards: dict[str, str] = {}
        page = 1
        while len(cards) < self.request.max_items:
            if not self._get(f"https://www.linkedin.com/search/results/people/?keywords={keyword}&page={page}"):
                break
            _scroll(self.driver, self.caution, times=1)
            found = self.driver.execute_script(_JS_PEOPLE) or {}
            new = {slug: text for slug, text in found.items() if slug not in cards and len(text) > 10}
            if not new:
                break
            cards.update(new)
            page += 1

        items = []
        for slug, card in list(cards.items())[: self.request.max_items]:
            lines = [line.strip() for line in card.split("\n") if line.strip()]
            # Carte type : nom, (degré), titre/headline, localisation, ...
            headline = next((line for line in lines[1:] if len(line) > 15 and "•" not in line), "")
            items.append(
                Item(
                    kind="person",
                    id=f"person:{slug}",
                    url=f"https://www.linkedin.com/in/{slug}/",
                    title=headline,
                    text=_clean(card),
                    keyword=self.request.keyword,
                    posted_at=self.now.date().isoformat(),
                    scraped_at=self.now.isoformat(),
                )
            )
        return items

    # ----------------------------------------------- profil utilisateur
    def own_profile_text(self) -> str:
        if not self._get("https://www.linkedin.com/in/me/"):
            return ""
        _scroll(self.driver, self.caution, times=2)
        main = _text(self.driver, ["main"])
        base = self.driver.current_url.split("?")[0].rstrip("/")
        skills = ""
        if "/in/" in base and self._get(f"{base}/details/skills/"):
            _scroll(self.driver, self.caution, times=2)
            skills = _text(self.driver, ["main"])
        return f"{main}\n\nCompétences :\n{skills}".strip()
