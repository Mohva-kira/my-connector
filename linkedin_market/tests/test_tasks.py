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
