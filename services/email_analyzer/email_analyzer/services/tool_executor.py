"""ToolExecutor — successeur de l'ancien ``email_analyzer/agent/executor.py``
(le package ``agent/`` est supprimé, voir progress-tracker.md). Diffère du
prédécesseur sur un point : les appels ne viennent plus d'un ``Planner``
déterministe à base de mots-clés mais des ``tool_calls`` natifs renvoyés par
le LLM (voir ``llm.stream_portfolio_assistant_chat_openai/_gemini``), et les
tools qui exigent un ``project_id`` (``summarize_project``, ``search_emails``,
``draft_reply``, et ``list_meetings`` quand il filtre) reçoivent désormais un
``project_name`` en langage naturel, résolu ici contre les projets réels du
tenant — ce que l'ancien ``Executor._extract_keyword`` (heuristique "premier
mot capitalisé") ne pouvait pas faire de façon fiable.

Mêmes garanties conservées de l'ancien Executor : Session SQLAlchemy fraîche
par appel d'outil (non thread-safe), exécution en parallèle
(ThreadPoolExecutor), isolation d'erreur par outil (un outil cassé ne casse
pas la boucle agentique — son erreur devient le contenu du tour ``tool``
renvoyé au LLM, qui peut s'adapter au lieu de faire planter la requête)."""

from __future__ import annotations

import difflib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type

from sqlalchemy.orm import Session, sessionmaker

from email_analyzer.classification import ProjectRules, rules_matrix_field_matches
from email_analyzer.db.models import Project
from email_analyzer.tools.draft_reply_tool import DraftReplyTool
from email_analyzer.tools.find_contact_tool import FindContactTool
from email_analyzer.tools.list_meetings_tool import ListMeetingsTool
from email_analyzer.tools.list_objectives_tool import ListObjectivesTool
from email_analyzer.tools.list_tickets_tool import ListTicketsTool
from email_analyzer.tools.search_documents_tool import SearchDocumentsTool
from email_analyzer.tools.search_emails_tool import SearchEmailsTool
from email_analyzer.tools.search_memory_tool import SearchMemoryTool
from email_analyzer.tools.search_project_summaries_tool import SearchProjectSummariesTool
from email_analyzer.tools.search_projects_tool import SearchProjectsTool
from email_analyzer.tools.summarize_project_tool import SummarizeProjectTool

logger = logging.getLogger(__name__)

# Format provider-agnostique consommé par llm.build_openai_tool_specs /
# llm._gemini_tools_param — un seul endroit décrit les paramètres exposés au
# LLM pour les 7 tools existants (email_analyzer/tools/, inchangés). Les
# tools qui portaient un `project_id: uuid.UUID` reçoivent ici un
# `project_name` en texte libre : voir `_resolve_project_id` ci-dessous pour
# la résolution. `tenant_id`/`user_id` ne font jamais partie de ce schéma —
# toujours injectés côté serveur, jamais choisis par le LLM.
TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "search_projects",
        "description": SearchProjectsTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Mot-clé optionnel à rechercher dans le nom du projet.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "search_memory",
        "description": SearchMemoryTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Mot-clé optionnel à rechercher dans le contenu des souvenirs.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "find_contact",
        "description": FindContactTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nom (ou partie du nom) du contact recherché.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "list_meetings",
        "description": ListMeetingsTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": (
                        "Nom du projet pour filtrer les rendez-vous (optionnel, "
                        "sinon tous les projets)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Nombre maximum de résultats (défaut 10).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_tickets",
        "description": ListTicketsTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": (
                        "Nom du projet pour filtrer les tickets (optionnel, sinon "
                        "tous les projets)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Nombre maximum de groupes de tickets (défaut 20).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_objectives",
        "description": ListObjectivesTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": (
                        "Nom du projet pour filtrer les objectifs (optionnel, sinon "
                        "tous les objectifs)."
                    ),
                }
            },
            "required": [],
        },
    },
    {
        "name": "summarize_project",
        "description": SummarizeProjectTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Nom exact ou approximatif du projet à résumer.",
                }
            },
            "required": ["project_name"],
        },
    },
    {
        "name": "search_project_summaries",
        "description": SearchProjectSummariesTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": (
                        "Nom exact ou approximatif du projet dont on veut l'historique "
                        "des résumés."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "Mot-clé optionnel à rechercher dans le contenu des résumés "
                        "archivés."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Nombre maximum de résumés archivés à retourner (défaut 10).",
                },
            },
            "required": ["project_name"],
        },
    },
    {
        "name": "search_emails",
        "description": SearchEmailsTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": (
                        "Nom exact ou approximatif du projet dont on veut les emails."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "Mot-clé optionnel à rechercher dans le sujet ou le corps des "
                        "emails du projet (sinon, les emails les plus importants sont "
                        "retournés)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Nombre maximum d'emails (défaut 10).",
                },
            },
            "required": ["project_name"],
        },
    },
    {
        "name": "search_documents",
        "description": SearchDocumentsTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Mot-clé optionnel à rechercher dans le nom ou le contenu des documents.",
                },
                "project_name": {
                    "type": "string",
                    "description": "Nom de projet pour filtrer les documents (optionnel, sinon tous les projets).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "draft_reply",
        "description": DraftReplyTool.description,
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": (
                        "Nom exact ou approximatif du projet pour lequel générer un "
                        "brouillon de réponse."
                    ),
                }
            },
            "required": ["project_name"],
        },
    },
]

# Tools dont run() attend un project_id (résolu ici depuis project_name) —
# list_meetings/list_tickets/search_documents/list_objectives y figurent
# aussi mais avec un project_name optionnel.
_PROJECT_NAME_TOOLS = {
    "summarize_project",
    "search_emails",
    "search_project_summaries",
    "draft_reply",
    "list_meetings",
    "list_tickets",
    "search_documents",
    "list_objectives",
}
_OPTIONAL_PROJECT_NAME_TOOLS = {"list_meetings", "list_tickets", "search_documents", "list_objectives"}


class ToolExecutor:
    def __init__(self, db: Session) -> None:
        # Classes, pas instances : chaque appel parallèle construit son
        # propre outil sur sa propre Session (voir _execute_one) plutôt que
        # de partager `db` entre threads (Session SQLAlchemy non thread-safe).
        self._tool_classes: Dict[str, Type[Any]] = {
            "search_projects": SearchProjectsTool,
            "summarize_project": SummarizeProjectTool,
            "search_emails": SearchEmailsTool,
            "search_project_summaries": SearchProjectSummariesTool,
            "search_memory": SearchMemoryTool,
            "find_contact": FindContactTool,
            "list_meetings": ListMeetingsTool,
            "list_tickets": ListTicketsTool,
            "list_objectives": ListObjectivesTool,
            "search_documents": SearchDocumentsTool,
            "draft_reply": DraftReplyTool,
        }
        self._session_factory = sessionmaker(bind=db.get_bind())

    def execute_calls(
        self, calls: List[Dict[str, Any]], tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """``calls`` : ``[{"id","name","arguments"}]`` tel qu'émis par
        l'événement ``tool_calls`` de ``llm.py``. Retourne, dans l'ordre
        d'appel (``ThreadPoolExecutor.map`` préserve l'ordre d'entrée), une
        liste ``[{"id","name","content","is_error"}]`` — ``content`` est le
        texte JSON à renvoyer au LLM comme tour ``tool``, ``is_error``
        permet à l'appelant (ConversationService) de distinguer un échec
        sans reparser ce JSON (utilisé pour le statut de l'événement SSE
        ``tool_result``)."""
        if not calls:
            return []
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            return list(pool.map(lambda call: self._execute_one(call, tenant_id, user_id), calls))

    def _execute_one(
        self, call: Dict[str, Any], tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> Dict[str, Any]:
        tool_name = call.get("name")
        call_id = call.get("id")
        arguments = call.get("arguments") or {}
        tool_cls = self._tool_classes.get(tool_name)
        if tool_cls is None:
            logger.warning("tool_executor unknown_tool tool=%s", tool_name)
            return {
                "id": call_id,
                "name": tool_name,
                "content": json.dumps({"error": "Outil inconnu"}, ensure_ascii=False),
                "is_error": True,
            }

        started = time.monotonic()
        logger.info("tool_executor tool_start tool=%s tenant=%s", tool_name, tenant_id)
        session = self._session_factory()
        is_error = False
        try:
            data = self._run_tool(session, tool_cls, tool_name, tenant_id, user_id, arguments)
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "tool_executor tool_done tool=%s tenant=%s duration_ms=%d",
                tool_name,
                tenant_id,
                duration_ms,
            )
            content = json.dumps({"data": data}, ensure_ascii=False, default=str)
        except Exception as ex:  # noqa: BLE001 — un outil cassé ne doit pas casser la boucle agentique
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "tool_executor tool_error tool=%s tenant=%s duration_ms=%d error=%s",
                tool_name,
                tenant_id,
                duration_ms,
                ex,
            )
            content = json.dumps({"error": str(ex)}, ensure_ascii=False)
            is_error = True
        finally:
            session.close()
        return {"id": call_id, "name": tool_name, "content": content, "is_error": is_error}

    def _run_tool(
        self,
        session: Session,
        tool_cls: Type[Any],
        tool_name: str,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        arguments: Dict[str, Any],
    ) -> Any:
        """Chaque tool a une signature légèrement différente (voir
        email_analyzer/tools/*.py) — mapping explicite ici plutôt que
        d'imposer une signature uniforme artificielle à des outils dont les
        entrées sont réellement différentes (un projet vs un tenant entier)."""
        tool = tool_cls(session)
        if tool_name in _PROJECT_NAME_TOOLS:
            project_name = arguments.get("project_name")
            project_id: Optional[uuid.UUID] = None
            # list_meetings/list_tickets/search_documents acceptent un
            # project_name optionnel (tous projets si absent) — les autres
            # l'exigent.
            if project_name or tool_name not in _OPTIONAL_PROJECT_NAME_TOOLS:
                project_id, error = self._resolve_project_id(session, tenant_id, project_name)
                if error:
                    raise ValueError(error)
            if tool_name == "list_meetings":
                return tool.run(tenant_id, project_id=project_id, limit=arguments.get("limit") or 10)
            if tool_name == "list_tickets":
                return tool.run(tenant_id, project_id=project_id, limit=arguments.get("limit") or 20)
            if tool_name == "search_documents":
                return tool.run(tenant_id, project_id=project_id, keyword=arguments.get("keyword"))
            if tool_name == "search_emails":
                return tool.run(
                    tenant_id,
                    project_id=project_id,
                    keyword=arguments.get("keyword"),
                    limit=arguments.get("limit") or 10,
                )
            if tool_name == "search_project_summaries":
                return tool.run(
                    tenant_id,
                    project_id=project_id,
                    keyword=arguments.get("keyword"),
                    limit=arguments.get("limit") or 10,
                )
            return tool.run(tenant_id, project_id=project_id)  # summarize_project, draft_reply, list_objectives
        if tool_name == "search_memory":
            return tool.run(tenant_id, user_id, keyword=arguments.get("keyword"))
        if tool_name == "search_projects":
            return tool.run(tenant_id, keyword=arguments.get("keyword"))
        if tool_name == "find_contact":
            return tool.run(tenant_id, name=arguments.get("name"))
        return tool.run(tenant_id)

    @staticmethod
    def _resolve_project_id(
        session: Session, tenant_id: uuid.UUID, project_name: Optional[str]
    ) -> Tuple[Optional[uuid.UUID], Optional[str]]:
        """Résout un nom de projet en langage naturel (tel que le LLM le
        recopie depuis le contexte système, voir ContextService) vers un
        ``project_id`` réel : 1) correspondance exacte insensible à la
        casse sur le nom, 2) correspondance exacte sur un champ de
        ``rules_matrix`` (domaine expéditeur, client... — voir
        ``rules_matrix_field_matches``, pour le cas où l'utilisateur désigne
        le projet par son domaine plutôt que son nom affiché), 3) sous-chaîne
        (nom OU rules_matrix) si un seul résultat, 4) correspondance floue
        (``difflib`` sur le nom, tolère une faute de frappe du LLM). Retourne
        ``(None, message_erreur_français)`` en cas d'absence ou d'ambiguïté —
        ce message devient le contenu du tour ``tool``, donc visible du LLM,
        qui peut alors redemander une précision ou réessayer avec un nom
        corrigé plutôt que de faire planter la requête (ce que l'ancien
        Planner déterministe ne pouvait jamais faire)."""
        if not project_name:
            return None, "Aucun nom de projet fourni."

        rows = (
            session.query(Project.id, Project.name, Project.rules_matrix)
            .filter(Project.tenant_id == tenant_id)
            .all()
        )
        if not rows:
            return None, "Aucun projet trouvé pour ce compte."

        needle = project_name.strip().lower()

        for project_id, name, _rules_matrix in rows:
            if name.strip().lower() == needle:
                return project_id, None

        exact_rules_matches = []
        for project_id, name, rules_matrix in rows:
            matches = rules_matrix_field_matches(ProjectRules.from_dict(rules_matrix), needle, exact=True)
            if matches:
                exact_rules_matches.append((project_id, name, matches))
        distinct_ids = {pid for pid, _name, _matches in exact_rules_matches}
        if len(distinct_ids) == 1:
            return exact_rules_matches[0][0], None
        if len(distinct_ids) > 1:
            labels = ", ".join(
                f"{name} ({label} « {value} »)"
                for _pid, name, matches in exact_rules_matches
                for label, value in matches[:1]
            )
            return (
                None,
                f"Plusieurs projets correspondent à « {project_name} » via leurs règles de "
                f"rattachement : {labels}. Précise lequel.",
            )

        substring_matches: Dict[uuid.UUID, str] = {}
        for project_id, name, rules_matrix in rows:
            if needle in name.lower() or rules_matrix_field_matches(
                ProjectRules.from_dict(rules_matrix), needle, exact=False
            ):
                substring_matches[project_id] = name
        if len(substring_matches) == 1:
            return next(iter(substring_matches)), None
        if len(substring_matches) > 1:
            names = ", ".join(substring_matches.values())
            return None, f"Plusieurs projets correspondent à « {project_name} » : {names}. Précise lequel."

        names_by_lower = {name.lower(): (pid, name) for pid, name, _rules_matrix in rows}
        close = difflib.get_close_matches(needle, list(names_by_lower.keys()), n=1, cutoff=0.6)
        if close:
            pid, _name = names_by_lower[close[0]]
            return pid, None

        return None, f"Aucun projet nommé « {project_name} » trouvé pour ce compte."
