import re
import socketserver
import threading

import pytest

from linkedin_market import config, contacts, outreach
from linkedin_market.tests.browser_fixtures import url_of

CAUTION = config.Caution("test", 1, 0.0, 0.0)


# ------------------------------------------------------------------- contacts
def test_parse_team_and_extract_email():
    rows = [
        {"slug": "jane-doe", "text": "Jane Doe • 2e\nTech Recruiter\nEnvoyer un message"},
        {"slug": "x", "text": ""},
    ]
    parsed = contacts.parse_team(rows)
    assert parsed[0] == {
        "full_name": "Jane Doe",
        "profile_url": "https://www.linkedin.com/in/jane-doe/",
        "email": None,
        "headline": "Tech Recruiter",
        "source": "linkedin_equipe_recrutement",
    }
    assert parsed[1]["full_name"] == "X"  # repli sur le slug
    assert contacts.extract_email(["bad", "a@b.io"], "") == "a@b.io"
    assert contacts.extract_email([], "Écris à jane@acme.com ou au 06") == "jane@acme.com"
    assert contacts.extract_email([], "rien") is None


def test_merge_keeps_manual_email():
    existing = [
        {
            "full_name": "Jane",
            "profile_url": "https://www.linkedin.com/in/jane-doe/",
            "email": "j@acme.com",
            "source": "manuel",
        },
        {"full_name": "Autre", "profile_url": "", "email": "o@x.io", "source": "manuel"},
    ]
    found = [
        {
            "full_name": "Jane Doe",
            "profile_url": "https://www.linkedin.com/in/jane-doe/",
            "email": None,
            "headline": "Recruiter",
            "source": "linkedin_equipe_recrutement",
        }
    ]
    merged = contacts.merge_contacts(existing, found)
    assert len(merged) == 2 and merged[0]["email"] == "j@acme.com" and merged[0]["full_name"] == "Jane Doe"


def test_find_hiring_team_on_page(driver):
    local = {
        "https://www.linkedin.com/in/jane-doe/": url_of("contact-jane.html"),
        "https://www.linkedin.com/in/john-roe/": url_of("contact-john.html"),
    }
    logs = []
    found = contacts.find_hiring_team(
        driver, url_of("hiring_team.html"), CAUTION, log_fn=logs.append, to_overlay=lambda u: local[u]
    )
    assert [c["full_name"] for c in found] == ["Jane Doe", "John Roe"]  # « Someone Else » (autre section) exclu
    assert found[0]["email"] == "jane@acme.com" and found[1]["email"] is None
    assert found[0]["headline"].startswith("Tech Recruiter")
    assert any("email non affiché" in line for line in logs)


def test_no_team_block_returns_nothing(driver):
    assert contacts.find_hiring_team(driver, url_of("external.html"), CAUTION, log_fn=lambda m: None) == []


# ----------------------------------------------------------------------- SMTP
class _SMTP(socketserver.StreamRequestHandler):
    def handle(self):
        self.server.mails = getattr(self.server, "mails", [])
        self.wfile.write(b"220 test\r\n")
        data, in_data = [], False
        for raw in self.rfile:
            line = raw.decode()
            if in_data:
                if line == ".\r\n":
                    self.server.mails.append("".join(data))
                    in_data = False
                    self.wfile.write(b"250 ok\r\n")
                else:
                    data.append(line)
                continue
            cmd = line[:4].upper()
            if cmd in ("EHLO", "HELO"):
                self.wfile.write(b"250 test\r\n")
            elif cmd in ("MAIL", "RCPT"):
                self.wfile.write(b"250 ok\r\n")
            elif cmd == "DATA":
                in_data = True
                self.wfile.write(b"354 go\r\n")
            elif cmd == "QUIT":
                self.wfile.write(b"221 bye\r\n")
                return
            else:
                self.wfile.write(b"250 ok\r\n")


@pytest.fixture
def smtp_server():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _SMTP)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_send_email_with_attachment(smtp_server, tmp_path):
    pdf = tmp_path / "cv.pdf"
    pdf.write_bytes(b"%PDF-1.4 x")
    smtp = {
        "host": "127.0.0.1",
        "port": smtp_server.server_address[1],
        "security": "none",
        "user": "",
        "password": "",
        "from": "me@example.com",
    }
    result = outreach.send_email("jane@acme.com", "Candidature — Dev", "Bonjour Jane\né", [pdf], smtp)
    assert result["to"] == "jane@acme.com" and result["attachments"] == ["cv.pdf"]
    raw = smtp_server.mails[0]
    assert "To: jane@acme.com" in raw and "Subject:" in raw and 'filename="cv.pdf"' in raw


def test_email_validation_and_missing_smtp(monkeypatch):
    for var in ("SMTP_HOST", "SMTP_FROM", "SMTP_USER"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(outreach.OutreachError, match="SMTP non configuré"):
        outreach.send_email("a@b.io", "s", "b")
    smtp = {"host": "127.0.0.1", "port": 9, "security": "none", "user": "", "password": "", "from": "me@example.com"}
    with pytest.raises(outreach.OutreachError, match="invalide"):
        outreach.send_email("pas un email", "s", "b", smtp=smtp)
    with pytest.raises(outreach.OutreachError, match="saut de ligne"):
        outreach.send_email("a@b.io", "s\nBcc: x@y.z", "b", smtp=smtp)
    with pytest.raises(outreach.OutreachError, match="Envoi impossible"):
        outreach.send_email("a@b.io", "s", "b", smtp=smtp)  # port fermé


def test_smtp_config_from_env(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.x.io")
    monkeypatch.setenv("SMTP_FROM", "me@x.io")
    monkeypatch.setenv("SMTP_SECURITY", "ssl")
    monkeypatch.delenv("SMTP_PORT", raising=False)
    assert config.smtp_config()["port"] == 465


# ------------------------------------------------------------------- LinkedIn
@pytest.fixture
def local_profiles(monkeypatch):
    monkeypatch.setattr(outreach, "_PROFILE_URL", re.compile(r"^file://"))


def test_linkedin_direct_message(driver, local_profiles):
    result = outreach.send_linkedin_message(
        driver, url_of("profile_first_degree.html"), "Bonjour Jane,\nvoici ma candidature.", CAUTION, timeout=5
    )
    assert result["mode"] == "message"
    assert driver.execute_script("return window.__sent") == "Bonjour Jane,\nvoici ma candidature."


def test_linkedin_connection_note_is_truncated(driver, local_profiles):
    result = outreach.send_linkedin_message(driver, url_of("profile_connect.html"), "x" * 400, CAUTION, timeout=5)
    assert result["mode"] == "connexion" and result["truncated"] is True
    assert len(driver.execute_script("return window.__note")) == outreach.LINKEDIN_NOTE_MAX


def test_linkedin_profile_without_message_or_connect(driver, local_profiles):
    with pytest.raises(outreach.OutreachError, match="InMail"):
        outreach.send_linkedin_message(driver, url_of("profile_none.html"), "Bonjour", CAUTION, timeout=2)


def test_linkedin_rejects_non_profile_urls():
    with pytest.raises(outreach.OutreachError, match="Profil LinkedIn invalide"):
        outreach.send_linkedin_message(None, "https://evil.example.com/in/x", "Bonjour", CAUTION)
