"""Tâches de fond de l'interface web (collecte, génération de CV, candidature, messages).

Chaque tâche tourne dans un thread. Les tâches qui pilotent le navigateur (``browser=True``)
se partagent un verrou : un seul Selenium à la fois, les autres restent « en_attente ».
Les logs du package ``linkedin_market`` émis par le thread de la tâche sont capturés pour
l'affichage en direct. Seuls les états finaux sont écrits dans ``tasks.jsonl``.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

from .. import config

PENDING, RUNNING, MANUAL, INPUT, DONE, FAILED, CANCELLED = (
    "en_attente", "en_cours", "attente_verification_manuelle", "attente_saisie", "terminée", "échec", "annulée")
FINAL = (DONE, FAILED, CANCELLED)
MAX_LOG_LINES = 400
INPUT_TIMEOUT_SECONDS = 600


class TaskCancelled(Exception):
    """Levée par ``TaskContext.check()`` quand l'utilisateur a demandé l'annulation."""


@dataclass
class Task:
    id: str
    kind: str
    title: str
    status: str = PENDING
    logs: list[str] = field(default_factory=list)
    result: Any = None
    error: str = ""
    needs_input: dict | None = None
    meta: dict = field(default_factory=dict)
    browser: bool = False
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""

    def public(self, with_logs: bool = True) -> dict:
        data = asdict(self)
        if not with_logs:
            data["logs"] = data["logs"][-5:]
        return data


class TaskContext:
    """Passé à la fonction de la tâche : journal, annulation, saisie utilisateur."""

    def __init__(self, manager: "TaskManager", task: Task):
        self._manager, self.task = manager, task
        self._cancel = threading.Event()
        self._answer: str | None = None
        self._answered = threading.Event()

    def log(self, message: str) -> None:
        self._manager._append_log(self.task, message)

    def should_stop(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise TaskCancelled("Tâche annulée.")

    def set_status(self, status: str) -> None:
        with self._manager._lock:
            self.task.status = status

    def manual_check(self, message: str) -> None:
        """LinkedIn réclame une vérification (captcha, 2FA) : l'utilisateur la fait dans le navigateur."""
        self.log(message)
        self.set_status(MANUAL)

    def ask(self, question: str, field_id: str = "", options: list[str] | None = None, *,
            suggestion: str = "", review: str = "") -> str:
        """Bloque jusqu'à la réponse de l'utilisateur (``POST /api/tasks/{id}/answer``).

        ``review`` : texte à relire avant une action irréversible (champ ``confirm_*``).
        """
        self._answered.clear()
        self._answer = None
        with self._manager._lock:
            self.task.status = INPUT
            self.task.needs_input = {"question": question, "field": field_id, "options": options or [],
                                     "suggestion": suggestion, "review": review}
        self.log(f"Question : {question}")
        deadline_reached = not self._answered.wait(INPUT_TIMEOUT_SECONDS)
        with self._manager._lock:
            self.task.needs_input = None
            self.task.status = RUNNING
        self.check()
        if deadline_reached or self._answer is None:
            raise TimeoutError("Aucune réponse reçue à temps : la tâche est interrompue.")
        return self._answer


class _TaskLogHandler(logging.Handler):
    """Copie les logs ``linkedin_market`` du thread d'une tâche dans son journal."""

    def __init__(self, manager: "TaskManager"):
        super().__init__(level=logging.INFO)
        self._manager = manager

    def emit(self, record: logging.LogRecord) -> None:
        ctx = self._manager._threads.get(record.thread)
        if ctx is not None:
            self._manager._append_log(ctx.task, record.getMessage())


class TaskManager:
    def __init__(self, max_workers: int = 4, persist: bool = True):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lm-task")
        self._lock = threading.RLock()
        self._browser_lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._contexts: dict[str, TaskContext] = {}
        self._threads: dict[int | None, TaskContext] = {}
        self._persist = persist
        handler = _TaskLogHandler(self)
        logging.getLogger("linkedin_market").addHandler(handler)
        logging.getLogger("linkedin_market").setLevel(logging.INFO)
        self._handler = handler
        if persist:
            self._load_history()

    # ------------------------------------------------------------ cycle
    def submit(self, kind: str, title: str, fn: Callable[[TaskContext], Any], *,
               browser: bool = False, meta: dict | None = None) -> Task:
        task = Task(id=uuid.uuid4().hex[:12], kind=kind, title=title, browser=browser, meta=meta or {},
                    created_at=datetime.now().isoformat(timespec="seconds"))
        ctx = TaskContext(self, task)
        with self._lock:
            self._tasks[task.id] = task
            self._contexts[task.id] = ctx
        self._pool.submit(self._run, task, ctx, fn)
        return task

    def _run(self, task: Task, ctx: TaskContext, fn: Callable[[TaskContext], Any]) -> None:
        acquired = False
        try:
            if task.browser:
                while not self._browser_lock.acquire(timeout=0.5):
                    ctx.check()
                acquired = True
            ctx.check()
            with self._lock:
                task.status = RUNNING
                task.started_at = datetime.now().isoformat(timespec="seconds")
            self._threads[threading.get_ident()] = ctx
            result = fn(ctx)
            with self._lock:
                task.result, task.status = result, DONE
        except TaskCancelled:
            with self._lock:
                task.status, task.error = CANCELLED, "Tâche annulée."
        except Exception as exc:  # noqa: BLE001 - toute erreur doit finir dans l'état de la tâche
            self._append_log(task, traceback.format_exc(limit=3))
            with self._lock:
                task.status, task.error = FAILED, str(exc) or exc.__class__.__name__
        finally:
            self._threads.pop(threading.get_ident(), None)
            if acquired:
                self._browser_lock.release()
            with self._lock:
                task.needs_input = None
                task.finished_at = datetime.now().isoformat(timespec="seconds")
            self._save_final(task)

    def cancel(self, task_id: str) -> bool:
        ctx = self._contexts.get(task_id)
        task = self._tasks.get(task_id)
        if ctx is None or task is None or task.status in FINAL:
            return False
        ctx._cancel.set()
        ctx._answered.set()  # débloque une attente de saisie
        return True

    def answer(self, task_id: str, value: str) -> bool:
        ctx, task = self._contexts.get(task_id), self._tasks.get(task_id)
        if ctx is None or task is None or task.status != INPUT:
            return False
        ctx._answer = value
        ctx._answered.set()
        return True

    # ----------------------------------------------------------- lecture
    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list(self, limit: int = 50) -> list[Task]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)[:limit]

    def active(self) -> list[Task]:
        with self._lock:
            return [t for t in self._tasks.values() if t.status not in FINAL]

    def has_active(self, kind: str, key: str | None = None) -> bool:
        return any(t.kind == kind and (key is None or t.meta.get("key") == key) for t in self.active())

    # ------------------------------------------------------- persistance
    def _append_log(self, task: Task, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            task.logs.append(f"{stamp} {message}")
            del task.logs[:-MAX_LOG_LINES]

    def _save_final(self, task: Task) -> None:
        if not self._persist:
            return
        try:
            config.ensure_data_dirs()
            with config.TASKS_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(task), ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def _load_history(self, limit: int = 100) -> None:
        if not config.TASKS_FILE.exists():
            return
        try:
            lines = config.TASKS_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
        except OSError:
            return
        for line in lines:
            try:
                data = json.loads(line)
                task = Task(**{k: v for k, v in data.items() if k in Task.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError):
                continue
            self._tasks[task.id] = task

    def shutdown(self) -> None:
        logging.getLogger("linkedin_market").removeHandler(self._handler)
        self._pool.shutdown(wait=False, cancel_futures=True)
