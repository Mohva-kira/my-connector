"""Analyse locale : sentiment, NER, risque, résumés (transformers / spaCy ou heuristiques)."""

import logging
import re
import threading
from typing import Dict, List, Optional

from .config import use_local_ml

# Plancher ajouté au score de règles selon le niveau de risque déjà calculé
# par calculate_risk_score pour le projet (signal LLM/heuristique existant,
# aucun appel supplémentaire) — voir EmailIntelligentAnalyzer.score_email_importance.
_IMPORTANCE_RISK_LEVEL_FLOOR = {
    "CRITIQUE": 40,
    "MODÉRÉ": 20,
    "FAIBLE": 0,
}


class EmailIntelligentAnalyzer:
    def __init__(self):
        """Initialisation des modèles IA avec fallback."""
        logging.info("Chargement des modèles IA...")

        self.sentiment_available = False
        self.summarizer_available = False
        self.nlp_available = False
        self.sentiment_analyzer = None
        self.summarizer = None
        self.nlp = None

        if use_local_ml():
            try:
                from transformers import pipeline

                try:
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=-1,
                    )
                    self.sentiment_available = True
                    logging.info("Modèle de sentiment chargé")
                except Exception as e:
                    logging.warning("Sentiment analyzer: %s", e)
                    self.sentiment_analyzer = None

                try:
                    self.summarizer = pipeline(
                        "summarization",
                        model="facebook/bart-large-cnn",
                        device=-1,
                        max_length=100,
                        min_length=20,
                    )
                    self.summarizer_available = True
                    logging.info("Modèle de résumé chargé")
                except Exception as e:
                    logging.warning("Summarizer: %s", e)
                    self.summarizer = None

            except ImportError as e:
                logging.error("Transformers non disponible: %s", e)
                self.sentiment_analyzer = None
                self.summarizer = None

            try:
                import spacy

                try:
                    self.nlp = spacy.load("fr_core_news_sm")
                    self.nlp_available = True
                    logging.info("Modèle NER français chargé")
                except Exception as e:
                    logging.warning("SpaCy NER français non disponible: %s", e)
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                        self.nlp_available = True
                        logging.info("Modèle NER anglais chargé (fallback)")
                    except Exception as e2:
                        logging.warning("SpaCy NER anglais non disponible: %s", e2)
                        try:
                            self.nlp = spacy.blank("fr")
                            self.nlp_available = True
                            logging.info(
                                "Modèle spaCy vide français chargé (pas de NER disponible)"
                            )
                        except Exception as e3:
                            logging.warning("Impossible de charger spaCy: %s", e3)
                            self.nlp = None
            except ImportError as e:
                logging.error("spaCy non disponible: %s", e)
                self.nlp = None
        else:
            logging.info(
                "EMAIL_ANALYZER_USE_LOCAL_ML désactivé : pas de chargement transformers/spaCy"
            )

        self.setup_basic_analyzers()

        logging.info(
            "Modèles chargés - Sentiment: %s, Résumé: %s, NER: %s",
            self.sentiment_available,
            self.summarizer_available,
            self.nlp_available,
        )

    def setup_basic_analyzers(self):
        """Configuration des analyseurs de base (sans ML)."""
        self.risk_keywords = {
            "urgent": 3,
            "critique": 3,
            "problème": 2,
            "retard": 2,
            "échéance": 2,
            "budget": 1,
            "dépassement": 3,
            "erreur": 2,
            "bug": 2,
            "bloqué": 3,
            "annulé": 3,
            "report": 2,
            "danger": 3,
            "alerte": 2,
            "attention": 1,
        }

        self.positive_keywords = [
            "réussi",
            "succès",
            "bien",
            "parfait",
            "excellent",
            "terminé",
            "livré",
            "validé",
            "approuvé",
            "satisfait",
            "content",
        ]

        self.negative_keywords = [
            "problème",
            "erreur",
            "échec",
            "retard",
            "bloqué",
            "difficile",
            "impossible",
            "mauvais",
            "inquiet",
            "préoccupé",
            "urgent",
        ]

        self.tech_keywords = [
            "react",
            "angular",
            "vue",
            "python",
            "java",
            "javascript",
            "nodejs",
            "api",
            "rest",
            "graphql",
            "mongodb",
            "mysql",
            "postgresql",
            "aws",
            "azure",
            "docker",
            "kubernetes",
            "jenkins",
            "git",
        ]

    def analyze_sentiment_basic(self, text: str) -> Dict:
        """Analyse de sentiment basique par mots-clés."""
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        if positive_count > negative_count:
            return {"label": "POSITIVE", "score": 0.6 + (positive_count * 0.1)}
        if negative_count > positive_count:
            return {"label": "NEGATIVE", "score": 0.6 + (negative_count * 0.1)}
        return {"label": "NEUTRAL", "score": 0.5}

    def analyze_sentiment(self, emails: List[Dict]) -> Dict:
        """Analyse de sentiment avec fallback."""
        sentiments = []

        for email_data in emails:
            try:
                normalized = email_data.get("normalized_text")
                if normalized:
                    text = normalized[:512]
                else:
                    text = f"{email_data['subject']} {email_data['body']}".lower()[:512]
                if not text.strip():
                    continue

                if self.sentiment_available and self.sentiment_analyzer:
                    try:
                        result = self.sentiment_analyzer(text)[0]
                        sentiment_result = {
                            "subject": email_data["subject"],
                            "sentiment": result["label"],
                            "confidence": result["score"],
                            "date": email_data["date"],
                            "method": "ML",
                        }
                    except Exception as e:
                        logging.warning("Erreur ML sentiment, fallback: %s", e)
                        basic_result = self.analyze_sentiment_basic(text)
                        sentiment_result = {
                            "subject": email_data["subject"],
                            "sentiment": basic_result["label"],
                            "confidence": basic_result["score"],
                            "date": email_data["date"],
                            "method": "Basic",
                        }
                else:
                    basic_result = self.analyze_sentiment_basic(text)
                    sentiment_result = {
                        "subject": email_data["subject"],
                        "sentiment": basic_result["label"],
                        "confidence": basic_result["score"],
                        "date": email_data["date"],
                        "method": "Basic",
                    }

                sentiments.append(sentiment_result)

            except Exception as e:
                logging.warning("Erreur analyse sentiment: %s", e)
                continue

        if sentiments:
            positive_count = sum(1 for s in sentiments if "POSITIVE" in s["sentiment"])
            negative_count = sum(1 for s in sentiments if "NEGATIVE" in s["sentiment"])
            neutral_count = len(sentiments) - positive_count - negative_count
            avg_confidence = sum(s["confidence"] for s in sentiments) / len(sentiments)

            return {
                "emails_analysés": len(sentiments),
                "sentiment_positif": positive_count,
                "sentiment_négatif": negative_count,
                "sentiment_neutre": neutral_count,
                "confiance_moyenne": round(avg_confidence, 2),
                "détails": sentiments[-3:],
                "tendance": (
                    "Positive"
                    if positive_count > negative_count
                    else "Négative"
                    if negative_count > positive_count
                    else "Neutre"
                ),
                "méthode": "ML" if self.sentiment_available else "Mots-clés",
            }

        return {"emails_analysés": 0}

    def extract_entities_basic(self, text: str) -> Dict:
        """Extraction d'entités basique par regex."""
        entities = {
            "emails": set(),
            "montants": set(),
            "dates": set(),
            "technologies": set(),
            "urls": set(),
        }

        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities["emails"].update(re.findall(email_pattern, text, re.IGNORECASE))

        montant_pattern = r"(\d+(?:\.\d{3})*(?:,\d{2})?)\s*(?:€|euros?|USD|\$|FCFA)"
        entities["montants"].update(re.findall(montant_pattern, text, re.IGNORECASE))

        url_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"
        entities["urls"].update(re.findall(url_pattern, text, re.IGNORECASE))

        text_lower = text.lower()
        for tech in self.tech_keywords:
            if tech in text_lower:
                entities["technologies"].add(tech.upper())

        return entities

    def extract_entities(self, emails: List[Dict]) -> Dict:
        """Extraction d'entités avec fallback."""
        all_entities = {
            "personnes": set(),
            "organisations": set(),
            "lieux": set(),
            "emails": set(),
            "montants": set(),
            "technologies": set(),
            "urls": set(),
        }

        for email_data in emails:
            try:
                text = email_data.get("normalized_text")
                if not text:
                    text = f"{email_data['subject']} {email_data['body']}".lower()

                if self.nlp_available and self.nlp:
                    try:
                        doc = self.nlp(text[:1000])
                        for ent in doc.ents:
                            if ent.label_ in ["PERSON", "PER"]:
                                all_entities["personnes"].add(ent.text)
                            elif ent.label_ in ["ORG", "ORGANIZATION"]:
                                all_entities["organisations"].add(ent.text)
                            elif ent.label_ in ["LOC", "LOCATION", "GPE"]:
                                all_entities["lieux"].add(ent.text)
                    except Exception as e:
                        logging.warning("Erreur spaCy, fallback: %s", e)

                basic_entities = self.extract_entities_basic(text)
                for key, values in basic_entities.items():
                    if key in all_entities:
                        all_entities[key].update(values)

            except Exception as e:
                logging.warning("Erreur extraction entités: %s", e)
                continue

        return {key: list(values)[:10] for key, values in all_entities.items()}

    def generate_auto_summary(self, emails: List[Dict]) -> Dict:
        """Génération de résumé avec fallback."""
        if not emails:
            return {"error": "Pas d'emails à résumer"}

        try:
            if self.summarizer_available and self.summarizer:
                try:
                    recent_emails = emails[-3:]
                    combined_text = ""

                    for email_data in recent_emails:
                        email_text = f"{email_data['subject']}. {email_data['body']}"
                        combined_text += email_text + " "

                    combined_text = combined_text[:800]

                    if len(combined_text.strip()) < 50:
                        return {"résumé": "Contenu insuffisant pour générer un résumé"}

                    summary_result = self.summarizer(combined_text)
                    summary_text = summary_result[0]["summary_text"]

                    return {
                        "résumé_automatique": summary_text,
                        "emails_analysés": len(recent_emails),
                        "méthode": "ML",
                        "taux_compression": round(len(summary_text) / len(combined_text), 2),
                    }
                except Exception as e:
                    logging.warning("Erreur ML résumé, fallback: %s", e)

            return self.generate_basic_summary(emails)

        except Exception as e:
            logging.error("Erreur génération résumé: %s", e)
            return {"error": f"Erreur résumé: {str(e)}"}

    def generate_basic_summary(self, emails: List[Dict]) -> Dict:
        """Résumé basique par extraction."""
        recent_emails = emails[-3:]
        key_sentences = []

        for email_data in recent_emails:
            subject = email_data["subject"]
            body = email_data["body"]

            if subject:
                key_sentences.append(f"Sujet: {subject}")

            if body:
                sentences = body.split(".")[:2]
                for sentence in sentences:
                    if len(sentence.strip()) > 20:
                        key_sentences.append(sentence.strip())

        summary = ". ".join(key_sentences[:5])

        return {
            "résumé_automatique": summary[:300] + "..." if len(summary) > 300 else summary,
            "emails_analysés": len(recent_emails),
            "méthode": "Extraction basique",
        }

    def calculate_risk_score(self, emails: List[Dict], sentiment_data: Dict, entities: Dict) -> Dict:
        """Calcul du score de risque."""
        risk_score = 0
        risk_factors = []

        try:
            if sentiment_data.get("tendance") == "Négative":
                risk_score += 30
                risk_factors.append("Sentiment négatif dominant")

            risk_keyword_count = 0
            for email_data in emails:
                text = email_data.get("normalized_text")
                if not text:
                    text = f"{email_data['subject']} {email_data['body']}".lower()
                for keyword, weight in self.risk_keywords.items():
                    if keyword in text:
                        risk_score += weight * 3
                        risk_keyword_count += 1

            if risk_keyword_count > 0:
                risk_factors.append(f"{risk_keyword_count} mots-clés de risque détectés")

            email_count = len(emails)
            if email_count > 50:
                risk_score += 10
                risk_factors.append("Volume d'emails élevé")
            elif email_count < 3:
                risk_score += 15
                risk_factors.append("Activité très faible")

            if risk_score >= 60:
                risk_level = "CRITIQUE"
            elif risk_score >= 30:
                risk_level = "MODÉRÉ"
            else:
                risk_level = "FAIBLE"

            return {
                "score_risque": min(risk_score, 100),
                "niveau_risque": risk_level,
                "facteurs_risque": risk_factors,
                "recommandation": self.get_recommendation(risk_level),
            }

        except Exception as e:
            logging.error("Erreur calcul risque: %s", e)
            return {"score_risque": 0, "niveau_risque": "INDETERMINÉ"}

    def get_recommendation(self, risk_level: str) -> str:
        """Recommandations basées sur le niveau de risque."""
        recommendations = {
            "CRITIQUE": "⚠️ ATTENTION IMMÉDIATE - Réviser le projet, contacter l'équipe",
            "MODÉRÉ": "⚡ Surveillance recommandée - Planifier un point d'équipe",
            "FAIBLE": "✅ Projet sur la bonne voie - Suivi normal",
        }
        return recommendations.get(risk_level, "Suivi standard recommandé")

    def identify_critical_emails(self, emails: List[Dict]) -> List[Dict]:
        """Identification des emails critiques."""
        critical_emails = []

        for email_data in emails:
            try:
                text = email_data.get("normalized_text")
                if not text:
                    text = f"{email_data['subject']} {email_data['body']}".lower()
                criticality_score = 0
                flags = []

                for keyword, weight in self.risk_keywords.items():
                    if keyword in text:
                        criticality_score += weight
                        flags.append(keyword)

                urgent_words = ["urgent", "asap", "immédiat", "critique", "emergency"]
                for word in urgent_words:
                    if word in text:
                        criticality_score += 5
                        flags.append(f"urgent_{word}")

                if criticality_score >= 4:
                    critical_emails.append(
                        {
                            "subject": email_data["subject"],
                            "from": email_data["from"],
                            "date": email_data["date"],
                            "criticality_score": criticality_score,
                            "flags": flags,
                            "preview": email_data["body"][:150] + "..."
                            if len(email_data["body"]) > 150
                            else email_data["body"],
                        }
                    )

            except Exception as e:
                logging.warning("Erreur analyse criticité: %s", e)
                continue

        critical_emails.sort(key=lambda x: x["criticality_score"], reverse=True)
        return critical_emails[:5]

    def score_email_importance(
        self,
        email_data: Dict,
        recipient_status: Optional[str] = None,
        niveau_risque: Optional[str] = None,
    ) -> int:
        """Score hybride d'importance 0-100 pour UN email, calculé au moment
        de la persistance (Fast-Track / sync planifiée — voir
        analysis_tasks.py) :

        - Règles : mêmes mots-clés pondérés que `identify_critical_emails`
          (`self.risk_keywords`), plafonnés puis mis à l'échelle (60 pts max) ;
          pas de nouvelle table de poids.
        - Adressage : +15 si l'email est adressé directement (``recipient_status
          == "direct_to"``), 0 si en copie (``"cc"``).
        - IA : plancher dérivé du `niveau_risque` déjà produit par l'appel LLM
          du même cycle de résumé (`_IMPORTANCE_RISK_LEVEL_FLOOR`) — pas
          d'appel LLM par email, coût/latence non maîtrisés à cette échelle.

        Volontairement pas de nouvel appel réseau : les trois signaux
        proviennent de calculs déjà faits ailleurs dans le pipeline.
        """
        text = email_data.get("normalized_text")
        if not text:
            text = f"{email_data.get('subject', '')} {email_data.get('body', '')}".lower()

        keyword_score = sum(weight for keyword, weight in self.risk_keywords.items() if keyword in text)
        rule_score = min(60, keyword_score * 3)

        recipient_bonus = 15 if recipient_status == "direct_to" else 0
        ai_floor = _IMPORTANCE_RISK_LEVEL_FLOOR.get(niveau_risque or "", 0)

        return min(100, rule_score + recipient_bonus + ai_floor)


_SHARED_ANALYZER: "EmailIntelligentAnalyzer | None" = None
_SHARED_ANALYZER_LOCK = threading.Lock()


def get_shared_analyzer() -> "EmailIntelligentAnalyzer":
    """Retourne un analyseur ML unique, partagé entre les requêtes.

    Les modèles transformers/spaCy sont coûteux à charger (dizaines de secondes
    sur CPU). Les instancier une seule fois évite un cold-load par requête, cause
    majeure des timeouts 504 sur ``/api/analyze``. L'inférence en lecture seule
    est sûre à partager entre les threads du threadpool.
    """
    global _SHARED_ANALYZER
    if _SHARED_ANALYZER is None:
        with _SHARED_ANALYZER_LOCK:
            if _SHARED_ANALYZER is None:
                _SHARED_ANALYZER = EmailIntelligentAnalyzer()
    return _SHARED_ANALYZER
