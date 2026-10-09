"""Parcours de l'interface compilée dans un vrai navigateur (Firefox headless) contre l'API réelle."""

import json
import socket
import threading
import time

import pytest
import uvicorn
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from linkedin_market import config, storage
from linkedin_market.tests.test_api import write_report
from linkedin_market.web.app import FRONTEND_DIST, create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = [
    pytest.mark.db,  # l'API lancée par le test lit réglages, historique et rapports dans la base de test
    pytest.mark.skipif(not (FRONTEND_DIST / "index.html").exists(), reason="frontend non compilé (npm run build)"),
]


@pytest.fixture
def base_url(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "profil.json").write_text(
        json.dumps({"title": "Dev", "skills": ["python"], "text": "Dev Python"}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    write_report()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    manager = TaskManager(persist=False)
    server = uvicorn.Server(uvicorn.Config(create_app(tasks=manager), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)
    manager.shutdown()


def wait(driver, condition, timeout=15):
    return WebDriverWait(driver, timeout).until(condition)


def test_skills_page_add_and_remove(driver, base_url):
    driver.get(f"{base_url}/competences")
    wait(driver, EC.presence_of_element_located((By.CSS_SELECTOR, "ul[aria-label='Compétences']")))
    field = driver.find_element(By.CSS_SELECTOR, "input[aria-label='Nouvelle compétence']")
    field.send_keys("Rust")
    driver.find_element(By.XPATH, "//button[normalize-space()='Ajouter']").click()
    wait(
        driver,
        EC.presence_of_element_located((By.XPATH, "//li[contains(., 'rust')]//button[@aria-label='Retirer rust']")),
    )
    wait(driver, lambda d: "Tes compétences ont changé" in d.page_source)  # scores périmés signalés
    driver.find_element(By.CSS_SELECTOR, "button[aria-label='Retirer rust']").click()
    wait(driver, lambda d: not d.find_elements(By.CSS_SELECTOR, "button[aria-label='Retirer rust']"))


def test_jobs_page_label_and_detail(driver, base_url):
    driver.get(f"{base_url}/offres")
    wait(driver, EC.presence_of_element_located((By.XPATH, "//button[normalize-space()='Senior Python Developer']")))
    driver.find_element(By.CSS_SELECTOR, "button[title='Pertinente']").click()
    wait(driver, lambda d: storage.load_labels().get("job:111") == 1)
    driver.find_element(By.XPATH, "//button[normalize-space()='Senior Python Developer']").click()
    wait(driver, EC.presence_of_element_located((By.XPATH, "//dialog[@open]//*[contains(., 'aws')]")))
    assert "Préparer la candidature" in driver.page_source


def test_dashboard_and_run_modal(driver, base_url):
    driver.get(base_url + "/")
    wait(driver, EC.presence_of_element_located((By.XPATH, "//h1[normalize-space()='Tableau de bord']")))
    wait(driver, lambda d: "Offres analysées" in d.page_source)
    driver.find_element(By.XPATH, "//button[normalize-space()='Lancer une analyse']").click()
    wait(driver, EC.presence_of_element_located((By.CSS_SELECTOR, "dialog[open] input")))
    assert "contraire aux conditions d'utilisation" in driver.page_source
