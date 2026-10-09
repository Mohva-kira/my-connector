import threading
import time

import pytest

from linkedin_market.web.tasks import CANCELLED, DONE, FAILED, INPUT, TaskManager

pytestmark = pytest.mark.db  # tâches persistées en PostgreSQL (vague 1)


def wait_for(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_success_failure_and_history():
    manager = TaskManager()
    ok = manager.submit("run", "ok", lambda ctx: {"value": 1})
    bad = manager.submit("run", "bad", lambda ctx: 1 / 0)
    assert wait_for(lambda: ok.status == DONE and bad.status == FAILED)
    assert ok.result == {"value": 1}
    assert "division" in bad.error
    reloaded = TaskManager()
    assert {t.id for t in reloaded.list()} >= {ok.id, bad.id}
    manager.shutdown(wait=True)
    reloaded.shutdown(wait=True)


def test_browser_tasks_run_one_at_a_time():
    manager = TaskManager()
    running, peak, lock = 0, 0, threading.Lock()

    def work(ctx):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.15)
        with lock:
            running -= 1

    tasks = [manager.submit("rescore", f"b{i}", work, browser=True) for i in range(3)]
    assert wait_for(lambda: all(t.status == DONE for t in tasks))
    assert peak == 1
    manager.shutdown(wait=True)


def test_cancel_while_running():
    manager = TaskManager()

    def loop(ctx):
        while True:
            ctx.check()
            time.sleep(0.02)

    task = manager.submit("run", "loop", loop)
    assert wait_for(lambda: task.status == "en_cours")
    assert manager.cancel(task.id)
    assert wait_for(lambda: task.status == CANCELLED)
    manager.shutdown(wait=True)


def test_ask_and_answer():
    manager = TaskManager()
    task = manager.submit("run", "ask", lambda ctx: ctx.ask("Années d'expérience ?", "years"))
    assert wait_for(lambda: task.status == INPUT)
    assert task.needs_input["question"].startswith("Années")
    assert manager.answer(task.id, "13")
    assert wait_for(lambda: task.status == DONE)
    assert task.result == "13"
    manager.shutdown(wait=True)


def test_logs_captured_only_for_own_thread():
    import logging

    manager = TaskManager()
    log = logging.getLogger("linkedin_market.scraper")

    def work(ctx):
        log.info("dans la tâche")
        ctx.log("direct")

    task = manager.submit("run", "log", work)
    assert wait_for(lambda: task.status == DONE)
    log.info("hors tâche")
    joined = "\n".join(task.logs)
    assert "dans la tâche" in joined and "direct" in joined and "hors tâche" not in joined
    manager.shutdown(wait=True)


def test_answer_removes_question_before_returning():
    """Une seule réponse par question : après ``answer()``, la question n'est plus visible (test déterministe)."""
    manager = TaskManager()
    gate = threading.Event()

    def two_questions(ctx):
        first = ctx.ask("Première ?", "q1")
        gate.wait(5)  # la tâche ne reprend la main qu'après les vérifications ci-dessous
        return first, ctx.ask("Seconde ?", "q2")

    task = manager.submit("run", "ask", two_questions)
    assert wait_for(lambda: task.status == INPUT)
    assert manager.answer(task.id, "1")
    assert task.needs_input is None and task.status != INPUT  # plus de question périmée à relire
    assert not manager.answer(task.id, "réponse en double")  # refusée, ne peut pas viser la question suivante
    gate.set()
    assert wait_for(lambda: task.status == INPUT and task.needs_input["field"] == "q2")
    assert manager.answer(task.id, "2")
    assert wait_for(lambda: task.status == DONE)
    assert task.result == ("1", "2")
    manager.shutdown(wait=True)


def test_on_abort_only_when_function_never_started():
    manager = TaskManager()
    gate, aborted = threading.Event(), []
    holder = manager.submit("rescore", "occupe", lambda ctx: gate.wait(5), browser=True)
    waiting = manager.submit(
        "message", "attend", lambda ctx: "envoyé", browser=True, on_abort=lambda: aborted.append(1)
    )
    assert wait_for(lambda: holder.status == "en_cours")
    assert manager.cancel(waiting.id)
    assert wait_for(lambda: waiting.status == CANCELLED)
    gate.set()
    ran = manager.submit("message", "démarre", lambda ctx: 1 / 0, on_abort=lambda: aborted.append(2))
    assert wait_for(lambda: ran.status == FAILED and holder.status == DONE)
    assert aborted == [1]  # rien pour une tâche dont la fonction a démarré, même en échec
    manager.shutdown(wait=True)
