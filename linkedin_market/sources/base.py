"""Briques communes aux sources de collecte : erreur, libellés et collecte par navigateur en lecture seule.

Couche intégrations : ce module n'importe ni FastAPI ni ``pipeline`` (l'annulation passe par la fonction ``check``
fournie par l'appelant, qui lève sa propre exception). Aucune écriture sur la plateforme (pas de candidature ni de
message automatisés, I-08, R-6), aucune technique de dissimulation : au premier blocage, la source s'arrête (I-06).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import quote_plus, urljoin, urlsplit

from selenium.common.exceptions import WebDriverException

from .. import config
from ..browser import human_pause
from ..models import Item, SearchRequest

log = logging.getLogger(__name__)

# Libellés des sources (``sources.label`` en base) : messages d'erreur, journaux et résultat de tâche.
LABELS = {
    "linkedin": "LinkedIn",
    "france_travail": "France Travail",
    "malt": "Malt",
    "wttj": "Welcome to the Jungle",
    "indeed": "Indeed",
}
# Préfixe de ``external_id`` par source (LinkedIn reste numérique, sans préfixe) : le ``ref`` ``job:<external_id>``
# ignore ``source_code`` (``find_item``), les identifiants doivent donc être uniques entre sources.
ID_PREFIXES = {"france_travail": "ft-", "malt": "malt-", "wttj": "wttj-", "indeed": "indeed-"}


def label_of(source_code: str) -> str:
    return LABELS.get(source_code, source_code)


def source_of_ref(ref: str) -> str:
    """Source d'origine déduite du préfixe d'un identifiant ``job:<external_id>`` (``linkedin`` par défaut)."""
    external_id = (ref or "").partition(":")[2]
    for code, prefix in ID_PREFIXES.items():
        if external_id.startswith(prefix):
            return code
    return "linkedin"


class SourceFetchError(RuntimeError):
    """Une source n'a pas pu être lue (réseau, quota, refus) : l'analyse continue avec les autres sources.

    Le message est en français, actionnable, et ne contient jamais de secret ni de jeton.
    """


# --------------------------------------------------------------------------- lecture du HTML
_SKIPPED_TAGS = {"script", "style", "noscript", "template", "svg"}
_BLOCK_TAGS = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr", "section", "article", "ul", "ol", "main"}


class Page:
    """Contenu utile d'une page (lecture de ``driver.page_source``) : liens, titres, texte visible, ``aria-label``.

    Les éléments sont repérés par leur balise sémantique (``h1``, ``main``, ``a``), leur ``href`` et leur
    ``aria-label`` : jamais par des classes CSS générées (R-5).
    """

    def __init__(self, html: str) -> None:
        parser = _PageParser()
        parser.feed(html or "")
        parser.close()
        self.links: list[tuple[str, str]] = parser.links  # (href, texte visible du lien)
        self.h1: str = _clean(parser.h1)
        self.title: str = _clean(parser.title)
        self.labelled: list[tuple[str, str]] = parser.labelled  # (aria-label en minuscules, texte)
        self.iframes: list[str] = parser.iframes
        self.main_text: str = _clean_block("".join(parser.main)) or _clean_block("".join(parser.body))
        self.body_text: str = _clean_block("".join(parser.body))

    def label(self, *names: str) -> str:
        """Texte de la première balise dont l'``aria-label`` (FR ou EN) contient l'un des ``names``."""
        for aria, text in self.labelled:
            if text and any(name in aria for name in names):
                return text
        return ""


_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class _Labelled:
    """Élément portant un ``aria-label`` en cours de lecture : balise, étiquette, texte, profondeur de même balise."""

    def __init__(self, tag: str, aria: str) -> None:
        self.tag, self.aria, self.parts, self.depth = tag, aria, [], 0


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.labelled: list[tuple[str, str]] = []
        self.iframes: list[str] = []
        self.h1 = ""
        self.title = ""
        self.main: list[str] = []
        self.body: list[str] = []
        self._skip = 0
        self._in_title = False
        self._in_h1 = False
        self._main_depth = 0
        self._link: list[str] | None = None
        self._link_href = ""
        self._open: list[_Labelled] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag in _SKIPPED_TAGS:
            self._skip += 1
            return
        if tag in _VOID_TAGS:
            if tag == "br":
                self._emit("\n")
            return
        if tag == "title":
            self._in_title = True
        elif tag == "h1" and not self.h1:
            self._in_h1 = True
        elif tag == "main":
            self._main_depth += 1
        elif tag == "iframe":
            self.iframes.append(values.get("src", ""))
        elif tag == "a" and values.get("href"):
            self._link, self._link_href = [], values["href"]
        if values.get("aria-label"):
            self._open.append(_Labelled(tag, values["aria-label"].lower()))
        elif self._open and self._open[-1].tag == tag:
            self._open[-1].depth += 1
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag in _VOID_TAGS:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False
        elif tag == "main":
            self._main_depth = max(0, self._main_depth - 1)
        elif tag == "a" and self._link is not None:
            self.links.append((self._link_href, _clean("".join(self._link))))
            self._link = None
        if self._open and self._open[-1].tag == tag:
            top = self._open[-1]
            if top.depth:
                top.depth -= 1
            else:
                self._open.pop()
                self.labelled.append((top.aria, _clean("".join(top.parts))))
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title += data
        if self._in_h1:
            self.h1 += data
        if self._link is not None:
            self._link.append(data)
        for element in self._open:
            element.parts.append(data)
        self._emit(data)

    def _emit(self, data: str) -> None:
        if self._skip:
            return
        self.body.append(data)
        if self._main_depth:
            self.main.append(data)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_block(text: str) -> str:
    lines = [_clean(line) for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


# ------------------------------------------------------------------------ collecte par navigateur
class BrowserJobSource:
    """Collecte des **offres** d'une plateforme en lecture seule, avec les règles de ``LinkedInScraper``.

    - ``check`` est appelé avant chaque chargement de page (annulation, E-22) ;
    - après un blocage (page de connexion, défi anti-robot, code 403 ou 429 affiché), plus aucune page n'est
      chargée (I-06) : ``blocked`` passe à vrai, aucune nouvelle tentative, aucun contournement ;
    - ``partial`` contient les offres déjà construites (conservées par l'appelant si la collecte est interrompue) ;
    - jamais de profil de personne (I-09) : les liens vers un profil sont ignorés, une page de profil n'est pas lue ;
    - ``max_items`` offres au plus, ``human_pause`` entre deux pages selon la prudence.

    Les classes filles fixent ``code``, ``search_url``, ``offer_pattern`` et ``person_markers``. Leurs adresses de
    recherche sont des constantes de module **non vérifiées en réel** (R-6) : une page non reconnue donne 0 élément
    et un avertissement, jamais une erreur.
    """

    code = ""
    # Adresse de recherche avec {keyword} (encodé) ; non vérifiée en réel.
    search_url = ""
    # Expression sur le chemin d'un lien d'offre ; le groupe nommé ``slug`` donne l'identifiant de l'offre.
    offer_pattern: re.Pattern[str] = re.compile(r"$^")
    # Fragments de chemin qui désignent un profil de personne : ces liens ne sont jamais ouverts (I-09).
    person_markers: tuple[str, ...] = ("/profile/", "/profil/", "/in/", "/freelance/", "/freelances/")
    # Chemins (ou fragments d'adresse) d'une page de connexion ou de vérification.
    login_markers: tuple[str, ...] = ("/login", "/signin", "/sign-in", "/connexion", "/challenge", "/captcha")
    # Titre ou texte court d'une page de blocage (comparés en minuscules).
    block_markers: tuple[str, ...] = (
        "captcha",
        "verify you are human",
        "vérifie que tu es un humain",
        "vérifiez que vous êtes un humain",
        "are you a robot",
        "just a moment",
        "un instant",
        "access denied",
        "accès refusé",
        "403 forbidden",
        "429 too many requests",
        "too many requests",
        "trop de requêtes",
        "unusual traffic",
        "trafic inhabituel",
    )
    iframe_markers: tuple[str, ...] = ("captcha", "challenges.cloudflare.com", "hcaptcha", "recaptcha")
    # Une page plus longue que cela est une vraie page de contenu, pas un écran de blocage.
    short_page_chars = 1500

    def __init__(self, driver, request: SearchRequest, *, check: Callable[[], None] | None = None) -> None:
        self.driver = driver
        self.request = request
        self.caution = config.CAUTION_LEVELS.get(request.caution, config.CAUTION_LEVELS[config.DEFAULT_CAUTION])
        self.now = datetime.now(UTC)
        self._check = check or (lambda: None)
        self.blocked = False
        self.partial: list[Item] = []
        self._page: Page | None = None

    # ------------------------------------------------------------ navigation
    def _is_blocked(self, page: Page) -> bool:
        url = (self.driver.current_url or "").lower()
        path = urlsplit(url).path
        if any(marker in path for marker in self.login_markers):
            return True
        title = page.title.lower()
        if any(marker in title for marker in self.block_markers):
            return True
        if any(marker in (src or "").lower() for src in page.iframes for marker in self.iframe_markers):
            return True
        if len(page.body_text) < self.short_page_chars:
            text = page.body_text.lower()
            return any(marker in text for marker in self.block_markers)
        return False

    def _get(self, url: str) -> Page | None:
        """Charge une page ; ``None`` après un blocage (aucune page n'est chargée ensuite) ou si elle est illisible."""
        if self.blocked:
            return None
        self._check()
        try:
            self.driver.get(url)
        except WebDriverException as exc:
            log.warning("%s : chargement impossible (%s).", label_of(self.code), exc.msg)
            return None
        human_pause(self.caution)
        page = Page(self.driver.page_source)
        if self._is_blocked(page):
            self.blocked = True
            log.error(
                "%s a demandé une connexion ou une vérification : arrêt de la collecte de cette source, "
                "repos de 6 h. Relance plus tard avec le niveau « prudent ».",
                label_of(self.code),
            )
            return None
        self._page = page
        return page

    # -------------------------------------------------------------- collecte
    def collect(self) -> list[Item]:
        """Offres de la recherche : une page de résultats, puis une page par offre (``max_items`` au plus)."""
        self.partial = []
        page = self._get(self.search_url.format(keyword=quote_plus(self.request.keyword)))
        if page is None:
            return []
        offers = self.offer_links(page)
        if not offers:
            log.warning(
                "%s : aucune offre reconnue sur la page de résultats (la structure du site a peut-être changé).",
                label_of(self.code),
            )
            return []
        offers = offers[: self.request.max_items]
        log.info("%s : %d offres trouvées, lecture du détail...", label_of(self.code), len(offers))
        for external_id, url in offers:
            detail = self._get(url)
            if detail is None:
                break
            item = self.build_item(external_id, url, detail)
            if item is not None:
                self.partial.append(item)
        return list(self.partial)

    def offer_links(self, page: Page) -> list[tuple[str, str]]:
        """``(external_id, url absolue)`` des offres de la page de résultats, sans doublon, jamais un profil (I-09)."""
        base = self.driver.current_url or self.search_url
        found: dict[str, str] = {}
        for href, _text in page.links:
            absolute = urljoin(base, href).split("#", 1)[0].split("?", 1)[0]
            path = urlsplit(absolute).path
            if any(marker in path.lower() for marker in self.person_markers):
                continue
            match = self.offer_pattern.search(path)
            if not match:
                continue
            slug = match["slug"]
            found.setdefault(f"{ID_PREFIXES[self.code]}{slug}", absolute)
        return list(found.items())

    def build_item(self, external_id: str, url: str, page: Page) -> Item | None:
        """Offre lue sur sa page de détail ; ``None`` si la page n'a pas de titre exploitable."""
        title = page.h1 or page.title
        if not title:
            log.warning("%s : page d'offre sans titre, ignorée.", label_of(self.code))
            return None
        text = page.main_text
        return Item(
            kind="job",
            id=f"job:{external_id}",
            url=url,
            title=_clean(title)[:150],
            text=text,
            keyword=self.request.keyword,
            company=self.company_of(url, page)[:100],
            location=page.label("lieu", "location", "localisation", "ville")[:120],
            posted_at=self.now.date().isoformat(),
            scraped_at=self.now.isoformat(),
            source_code=self.code,
        )

    def company_of(self, url: str, page: Page) -> str:
        """Entreprise : texte du lien d'entreprise ou ``aria-label`` « Entreprise »/« Company » de la page."""
        return page.label("entreprise", "company", "société")
