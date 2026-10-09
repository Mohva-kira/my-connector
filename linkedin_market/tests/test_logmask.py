"""Masquage des données de tiers dans les journaux (G-9, E-12)."""

from linkedin_market import logmask


def test_adresse_nom_et_profil():
    assert logmask.mask_email("jane.doe@acme.com") == "j***@acme.com"
    assert logmask.mask_email("pas-un-mail") == "***" and logmask.mask_email("") == ""
    assert logmask.mask_name("Jane Doe") == "J. D."
    assert logmask.mask_name("Marie-Claire de la Tour") == "M. C. D. L. T."
    assert logmask.mask_name("") == ""
    assert logmask.mask_profile_url("https://www.linkedin.com/in/jane-doe/") == "linkedin.com/in/j***"
    assert logmask.mask_recipient("jane@acme.com") == "j***@acme.com"
    assert logmask.mask_recipient("https://fr.linkedin.com/in/jane-doe?x=1") == "linkedin.com/in/j***"


def test_texte_libre_idempotent():
    text = "Envoi à jane.doe@acme.com (https://www.linkedin.com/in/jane-doe/?trk=x) et john@x.io."
    masked = logmask.mask_text(text)
    assert masked == "Envoi à j***@acme.com (linkedin.com/in/j***) et j***@x.io."
    assert logmask.mask_text(masked) == masked
    assert logmask.mask_text("") == "" and logmask.mask_text("Étape 2 remplie.") == "Étape 2 remplie."


def test_expression_email_partagee():
    assert logmask.EMAIL_ADDRESS.match("me@example.com")
    assert not logmask.EMAIL_ADDRESS.match("a@b.io, c@d.io")
    assert not logmask.EMAIL_ADDRESS.match("<a@b.io>")
