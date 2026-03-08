"""
Script d'analyse d'emails de projets avec intelligence artificielle
Version corrigée pour compatibilité Keras/TensorFlow
"""

import imaplib
import email
from email.header import decode_header
import re
import logging
import time
from time import perf_counter
from datetime import datetime, timedelta
from collections import defaultdict
import argparse
from typing import List, Dict
import json
import os

# Configuration pour éviter les conflits
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_KERAS'] = '1'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmailIntelligentAnalyzer:
    def __init__(self):
        """Initialisation des modèles IA avec fallback"""
        logging.info("Chargement des modèles IA...")
        
        # Flags pour modèles disponibles
        self.sentiment_available = False
        self.summarizer_available = False
        self.nlp_available = False
        
        try:
            # Tentative de chargement des transformers
            from transformers import pipeline
            
            # Modèle de sentiment français léger
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=-1  # Force CPU
                )
                self.sentiment_available = True
                logging.info("✅ Modèle de sentiment chargé")
            except Exception as e:
                logging.warning(f"❌ Sentiment analyzer: {e}")
                self.sentiment_analyzer = None
            
            # Modèle de résumé avec fallback
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1,
                    max_length=100,
                    min_length=20
                )
                self.summarizer_available = True
                logging.info("✅ Modèle de résumé chargé")
            except Exception as e:
                logging.warning(f"❌ Summarizer: {e}")
                self.summarizer = None
                
        except ImportError as e:
            logging.error(f"Transformers non disponible: {e}")
            self.sentiment_analyzer = None
            self.summarizer = None
        
        # Modèle NER avec spaCy (plus stable)
        try:
            import spacy
            self.nlp = spacy.load("fr_core_news_sm")
            self.nlp_available = True
            logging.info("✅ Modèle NER français chargé")
        except Exception as e:
            logging.warning(f"❌ SpaCy NER: {e}")
            try:
                # Fallback vers modèle anglais
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp_available = True
                logging.info("✅ Modèle NER anglais chargé (fallback)")
            except:
                self.nlp = None
        
        # Modèles de base toujours disponibles
        self.setup_basic_analyzers()
        
        logging.info(f"Modèles chargés - Sentiment: {self.sentiment_available}, Résumé: {self.summarizer_available}, NER: {self.nlp_available}")

    def setup_basic_analyzers(self):
        """Configuration des analyseurs de base (sans ML)"""
        # Mots-clés de risque pour projets
        self.risk_keywords = {
            'urgent': 3, 'critique': 3, 'problème': 2, 'retard': 2, 'échéance': 2,
            'budget': 1, 'dépassement': 3, 'erreur': 2, 'bug': 2, 'bloqué': 3,
            'annulé': 3, 'report': 2, 'danger': 3, 'alerte': 2, 'attention': 1
        }
        
        # Mots-clés positifs/négatifs pour sentiment basic
        self.positive_keywords = [
            'réussi', 'succès', 'bien', 'parfait', 'excellent', 'terminé',
            'livré', 'validé', 'approuvé', 'satisfait', 'content'
        ]
        
        self.negative_keywords = [
            'problème', 'erreur', 'échec', 'retard', 'bloqué', 'difficile',
            'impossible', 'mauvais', 'inquiet', 'préoccupé', 'urgent'
        ]
        
        # Technologies à détecter
        self.tech_keywords = [
            'react', 'angular', 'vue', 'python', 'java', 'javascript', 'nodejs',
            'api', 'rest', 'graphql', 'mongodb', 'mysql', 'postgresql',
            'aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'git'
        ]

    def analyze_sentiment_basic(self, text: str) -> Dict:
        """Analyse de sentiment basique par mots-clés"""
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'POSITIVE', 'score': 0.6 + (positive_count * 0.1)}
        elif negative_count > positive_count:
            return {'label': 'NEGATIVE', 'score': 0.6 + (negative_count * 0.1)}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}

    def analyze_sentiment(self, emails: List[Dict]) -> Dict:
        """Analyse de sentiment avec fallback"""
        sentiments = []
        
        for email_data in emails:
            try:
                normalized = email_data.get('normalized_text')
                if normalized:
                    text = normalized[:512]
                else:
                    text = f"{email_data['subject']} {email_data['body']}".lower()[:512]
                if not text.strip():
                    continue
                
                # Essayer le modèle ML d'abord
                if self.sentiment_available and self.sentiment_analyzer:
                    try:
                        result = self.sentiment_analyzer(text)[0]
                        sentiment_result = {
                            'subject': email_data['subject'],
                            'sentiment': result['label'],
                            'confidence': result['score'],
                            'date': email_data['date'],
                            'method': 'ML'
                        }
                    except Exception as e:
                        logging.warning(f"Erreur ML sentiment, fallback: {e}")
                        basic_result = self.analyze_sentiment_basic(text)
                        sentiment_result = {
                            'subject': email_data['subject'],
                            'sentiment': basic_result['label'],
                            'confidence': basic_result['score'],
                            'date': email_data['date'],
                            'method': 'Basic'
                        }
                else:
                    # Utiliser l'analyse basique
                    basic_result = self.analyze_sentiment_basic(text)
                    sentiment_result = {
                        'subject': email_data['subject'],
                        'sentiment': basic_result['label'],
                        'confidence': basic_result['score'],
                        'date': email_data['date'],
                        'method': 'Basic'
                    }
                
                sentiments.append(sentiment_result)
                
            except Exception as e:
                logging.warning(f"Erreur analyse sentiment: {e}")
                continue
        
        # Statistiques globales
        if sentiments:
            positive_count = sum(1 for s in sentiments if 'POSITIVE' in s['sentiment'])
            negative_count = sum(1 for s in sentiments if 'NEGATIVE' in s['sentiment'])
            neutral_count = len(sentiments) - positive_count - negative_count
            avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            
            return {
                'emails_analysés': len(sentiments),
                'sentiment_positif': positive_count,
                'sentiment_négatif': negative_count,
                'sentiment_neutre': neutral_count,
                'confiance_moyenne': round(avg_confidence, 2),
                'détails': sentiments[-3:],  # 3 derniers
                'tendance': 'Positive' if positive_count > negative_count else 'Négative' if negative_count > positive_count else 'Neutre',
                'méthode': 'ML' if self.sentiment_available else 'Mots-clés'
            }
        
        return {"emails_analysés": 0}

    def extract_entities_basic(self, text: str) -> Dict:
        """Extraction d'entités basique par regex"""
        entities = {
            'emails': set(),
            'montants': set(),
            'dates': set(),
            'technologies': set(),
            'urls': set()
        }
        
        # Emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'].update(re.findall(email_pattern, text, re.IGNORECASE))
        
        # Montants
        montant_pattern = r'(\d+(?:\.\d{3})*(?:,\d{2})?)\s*(?:€|euros?|USD|\$|FCFA)'
        entities['montants'].update(re.findall(montant_pattern, text, re.IGNORECASE))
        
        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        entities['urls'].update(re.findall(url_pattern, text, re.IGNORECASE))
        
        # Technologies
        text_lower = text.lower()
        for tech in self.tech_keywords:
            if tech in text_lower:
                entities['technologies'].add(tech.upper())
        
        return entities

    def extract_entities(self, emails: List[Dict]) -> Dict:
        """Extraction d'entités avec fallback"""
        all_entities = {
            'personnes': set(),
            'organisations': set(),
            'lieux': set(),
            'emails': set(),
            'montants': set(),
            'technologies': set(),
            'urls': set()
        }
        
        for email_data in emails:
            try:
                text = email_data.get('normalized_text')
                if not text:
                    text = f"{email_data['subject']} {email_data['body']}".lower()
                
                # Essayer spaCy d'abord
                if self.nlp_available and self.nlp:
                    try:
                        doc = self.nlp(text[:1000])  # Limiter pour performance
                        for ent in doc.ents:
                            if ent.label_ in ["PERSON", "PER"]:
                                all_entities['personnes'].add(ent.text)
                            elif ent.label_ in ["ORG", "ORGANIZATION"]:
                                all_entities['organisations'].add(ent.text)
                            elif ent.label_ in ["LOC", "LOCATION", "GPE"]:
                                all_entities['lieux'].add(ent.text)
                    except Exception as e:
                        logging.warning(f"Erreur spaCy, fallback: {e}")
                
                # Extraction basique en complément
                basic_entities = self.extract_entities_basic(text)
                for key, values in basic_entities.items():
                    if key in all_entities:
                        all_entities[key].update(values)
                        
            except Exception as e:
                logging.warning(f"Erreur extraction entités: {e}")
                continue
        
        # Conversion en listes pour JSON
        return {
            key: list(values)[:10] for key, values in all_entities.items()
        }

    def generate_auto_summary(self, emails: List[Dict]) -> Dict:
        """Génération de résumé avec fallback"""
        if not emails:
            return {"error": "Pas d'emails à résumer"}
        
        try:
            # Essayer le modèle ML
            if self.summarizer_available and self.summarizer:
                try:
                    recent_emails = emails[-3:]
                    combined_text = ""
                    
                    for email_data in recent_emails:
                        email_text = f"{email_data['subject']}. {email_data['body']}"
                        combined_text += email_text + " "
                    
                    combined_text = combined_text[:800]  # Limiter
                    
                    if len(combined_text.strip()) < 50:
                        return {"résumé": "Contenu insuffisant pour générer un résumé"}
                    
                    summary_result = self.summarizer(combined_text)
                    summary_text = summary_result[0]['summary_text']
                    
                    return {
                        "résumé_automatique": summary_text,
                        "emails_analysés": len(recent_emails),
                        "méthode": "ML",
                        "taux_compression": round(len(summary_text) / len(combined_text), 2)
                    }
                except Exception as e:
                    logging.warning(f"Erreur ML résumé, fallback: {e}")
            
            # Résumé basique par extraction des premières phrases
            return self.generate_basic_summary(emails)
            
        except Exception as e:
            logging.error(f"Erreur génération résumé: {e}")
            return {"error": f"Erreur résumé: {str(e)}"}

    def generate_basic_summary(self, emails: List[Dict]) -> Dict:
        """Résumé basique par extraction"""
        recent_emails = emails[-3:]
        key_sentences = []
        
        for email_data in recent_emails:
            subject = email_data['subject']
            body = email_data['body']
            
            # Ajouter le sujet
            if subject:
                key_sentences.append(f"Sujet: {subject}")
            
            # Extraire la première phrase du corps
            if body:
                sentences = body.split('.')[:2]  # 2 premières phrases
                for sentence in sentences:
                    if len(sentence.strip()) > 20:
                        key_sentences.append(sentence.strip())
        
        summary = ". ".join(key_sentences[:5])  # Max 5 éléments
        
        return {
            "résumé_automatique": summary[:300] + "..." if len(summary) > 300 else summary,
            "emails_analysés": len(recent_emails),
            "méthode": "Extraction basique"
        }

    def calculate_risk_score(self, emails: List[Dict], sentiment_data: Dict, entities: Dict) -> Dict:
        """Calcul du score de risque"""
        risk_score = 0
        risk_factors = []
        
        try:
            # Facteur sentiment
            if sentiment_data.get('tendance') == 'Négative':
                risk_score += 30
                risk_factors.append("Sentiment négatif dominant")
            
            # Facteur mots-clés de risque
            risk_keyword_count = 0
            for email_data in emails:
                text = email_data.get('normalized_text')
                if not text:
                    text = f"{email_data['subject']} {email_data['body']}".lower()
                for keyword, weight in self.risk_keywords.items():
                    if keyword in text:
                        risk_score += weight * 3
                        risk_keyword_count += 1
            
            if risk_keyword_count > 0:
                risk_factors.append(f"{risk_keyword_count} mots-clés de risque détectés")
            
            # Facteur volume
            email_count = len(emails)
            if email_count > 50:
                risk_score += 10
                risk_factors.append("Volume d'emails élevé")
            elif email_count < 3:
                risk_score += 15
                risk_factors.append("Activité très faible")
            
            # Classification du risque
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
                "recommandation": self.get_recommendation(risk_level)
            }
            
        except Exception as e:
            logging.error(f"Erreur calcul risque: {e}")
            return {"score_risque": 0, "niveau_risque": "INDETERMINÉ"}

    def get_recommendation(self, risk_level: str) -> str:
        """Recommandations basées sur le niveau de risque"""
        recommendations = {
            "CRITIQUE": "⚠️ ATTENTION IMMÉDIATE - Réviser le projet, contacter l'équipe",
            "MODÉRÉ": "⚡ Surveillance recommandée - Planifier un point d'équipe",
            "FAIBLE": "✅ Projet sur la bonne voie - Suivi normal"
        }
        return recommendations.get(risk_level, "Suivi standard recommandé")

    def identify_critical_emails(self, emails: List[Dict]) -> List[Dict]:
        """Identification des emails critiques"""
        critical_emails = []
        
        for email_data in emails:
            try:
                text = email_data.get('normalized_text')
                if not text:
                    text = f"{email_data['subject']} {email_data['body']}".lower()
                criticality_score = 0
                flags = []
                
                # Mots-clés critiques
                for keyword, weight in self.risk_keywords.items():
                    if keyword in text:
                        criticality_score += weight
                        flags.append(keyword)
                
                # Présence d'urgence
                urgent_words = ['urgent', 'asap', 'immédiat', 'critique', 'emergency']
                for word in urgent_words:
                    if word in text:
                        criticality_score += 5
                        flags.append(f"urgent_{word}")
                
                if criticality_score >= 4:  # Seuil abaissé
                    critical_emails.append({
                        'subject': email_data['subject'],
                        'from': email_data['from'],
                        'date': email_data['date'],
                        'criticality_score': criticality_score,
                        'flags': flags,
                        'preview': email_data['body'][:150] + "..." if len(email_data['body']) > 150 else email_data['body']
                    })
                    
            except Exception as e:
                logging.warning(f"Erreur analyse criticité: {e}")
                continue
        
        critical_emails.sort(key=lambda x: x['criticality_score'], reverse=True)
        return critical_emails[:5]

# Reprise du reste du code EmailProjectAnalyzer...
class EmailProjectAnalyzer:
    def __init__(
        self,
        email_address: str,
        password: str,
        imap_server: str = "mail.mediasoftci.net",
        port: int = 993,
        max_deep_emails: int = 20,
        cache_file: str = "email_analysis_cache.json"
    ):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.port = port
        self.mail = None
        self.project_emails = defaultdict(list)
        self.max_deep_emails = max(5, max_deep_emails)
        self.cache_file = cache_file
        self.email_cache = self.load_cache()
        self.step_timings = defaultdict(float)
        self.step_counts = defaultdict(int)
        
        # Initialiser l'analyseur IA
        self.ai_analyzer = EmailIntelligentAnalyzer()

    def record_timing(self, step_name: str, duration_seconds: float):
        """Enregistre le temps passé sur une étape."""
        self.step_timings[step_name] += duration_seconds
        self.step_counts[step_name] += 1

    def get_top_timing_steps(self, top_n: int = 3) -> List[Dict]:
        """Retourne les étapes les plus coûteuses."""
        sorted_steps = sorted(self.step_timings.items(), key=lambda item: item[1], reverse=True)
        top_steps = []
        for step_name, total_seconds in sorted_steps[:top_n]:
            runs = self.step_counts.get(step_name, 1)
            top_steps.append({
                "étape": step_name,
                "total_s": round(total_seconds, 3),
                "runs": runs,
                "moyenne_s": round(total_seconds / max(1, runs), 3)
            })
        return top_steps

    def load_cache(self) -> Dict:
        """Charge le cache local des emails déjà traités."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    raw_cache = json.load(f)
                if isinstance(raw_cache, dict):
                    return raw_cache
        except Exception as e:
            logging.warning(f"Impossible de charger le cache: {e}")
        return {}

    def save_cache(self):
        """Sauvegarde un cache borné pour éviter les retraitements."""
        try:
            max_entries = 2000
            if len(self.email_cache) > max_entries:
                keys = list(self.email_cache.keys())[-max_entries:]
                self.email_cache = {k: self.email_cache[k] for k in keys}
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.email_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Impossible de sauvegarder le cache: {e}")

    def truncate_text(self, text: str, max_len: int = 160) -> str:
        """Tronque proprement le texte pour l'affichage console."""
        if not text:
            return ""
        clean = " ".join(text.split())
        if len(clean) <= max_len:
            return clean
        return clean[:max_len - 3].rstrip() + "..."

    def decode_best_effort(self, payload: bytes, charset_hint: str = None) -> str:
        """Décode au mieux les corps d'email pour limiter la perte d'accents."""
        if not payload:
            return ""
        candidates = [charset_hint, "utf-8", "latin-1", "cp1252"]
        for encoding in candidates:
            if not encoding:
                continue
            try:
                return payload.decode(encoding, errors='ignore')
            except Exception:
                continue
        return payload.decode("utf-8", errors='ignore')

    def normalize_email_text(self, email_content: Dict) -> str:
        """Construit une version normalisée réutilisable par les analyseurs."""
        subject = email_content.get('subject', '')
        body = email_content.get('body', '')
        return f"{subject} {body}".lower()

    def should_fetch_full_message(self, subject: str, project_filters: List[str]) -> bool:
        """Filtrage précoce: ne charge le corps complet que si le sujet est potentiellement pertinent."""
        subject_lower = (subject or "").lower()
        for project_filter in project_filters:
            if project_filter.lower() in subject_lower:
                return True
        return False

    def format_flags(self, flags: List[str]) -> str:
        """Rend les flags de criticité plus lisibles pour l'utilisateur."""
        if not flags:
            return "aucun indicateur explicite"
        readable = []
        for flag in flags[:3]:
            if flag.startswith("urgent_"):
                readable.append(f"urgence ({flag.replace('urgent_', '')})")
            else:
                readable.append(flag)
        return ", ".join(readable)

    def build_project_diagnostic(self, analysis: Dict) -> str:
        """Crée une synthèse orientée décision pour le rapport."""
        risk = analysis.get('évaluation_risque', {})
        sentiment = analysis.get('analyse_sentiment', {})
        priority = analysis.get('priorité_attention', 'NORMALE')
        risk_level = risk.get('niveau_risque', 'INDETERMINÉ')
        trend = sentiment.get('tendance', 'Neutre')
        email_count = analysis.get('nb_emails', 0)

        if risk_level == "CRITIQUE":
            return (
                f"Situation tendue ({email_count} emails, tendance {trend.lower()}). "
                f"Priorité {priority.lower()}: escalade immédiate et plan d'action court terme."
            )
        if risk_level == "MODÉRÉ":
            return (
                f"Situation sous surveillance ({email_count} emails, tendance {trend.lower()}). "
                f"Prévoir un point de cadrage et un suivi hebdomadaire."
            )
        return (
            f"Situation globalement stable ({email_count} emails, tendance {trend.lower()}). "
            f"Maintenir le rythme de suivi actuel."
        )

    def format_project_report(self, project_name: str, analysis: Dict) -> List[str]:
        """Formate un bloc de rapport projet plus narratif et actionnable."""
        lines = []
        lines.append(f"\n🚀 PROJET: {project_name}")
        lines.append(f"   📊 Priorité: {analysis.get('priorité_attention', 'N/A')}")
        lines.append(f"   📧 Emails: {analysis.get('nb_emails', 0)}")
        lines.append(f"   👥 Participants: {analysis.get('nb_participants', 0)}")
        lines.append(f"   🧭 Diagnostic: {self.build_project_diagnostic(analysis)}")

        sentiment = analysis.get('analyse_sentiment', {})
        if sentiment.get('tendance'):
            method = sentiment.get('méthode', 'N/A')
            confidence = sentiment.get('confiance_moyenne')
            confidence_txt = f", confiance moyenne {confidence}" if confidence is not None else ""
            lines.append(
                f"   😊 Sentiment: {sentiment['tendance']} (méthode: {method}{confidence_txt})"
            )

        risk = analysis.get('évaluation_risque', {})
        risk_score = risk.get('score_risque', 0)
        lines.append(f"   ⚠️ Risque: {risk.get('niveau_risque', 'N/A')} (score: {risk_score}/100)")
        risk_factors = risk.get('facteurs_risque', [])
        if risk_factors:
            lines.append(f"   🔎 Facteurs: {self.truncate_text('; '.join(risk_factors), 180)}")
        lines.append(f"   💡 Action: {risk.get('recommandation', 'Suivi standard recommandé')}")

        entities = analysis.get('entités_extraites', {})
        techs = entities.get('technologies', [])
        montants = entities.get('montants', [])
        if techs:
            lines.append(f"   💻 Technologies mentionnées: {', '.join(techs[:5])}")
        clean_montants = [m for m in montants if str(m).strip() and str(m).strip() != "000"]
        if clean_montants:
            lines.append(f"   💰 Montants détectés: {', '.join(clean_montants[:3])}")

        auto_summary = analysis.get('résumé_automatique', {})
        if isinstance(auto_summary, dict) and auto_summary.get('résumé_automatique'):
            method = auto_summary.get('méthode', 'N/A')
            summary_text = self.truncate_text(auto_summary['résumé_automatique'], 260)
            lines.append(f"   📝 Synthèse ({method}): {summary_text}")

        critical = analysis.get('emails_critiques', [])
        if critical:
            lines.append(f"   🚨 Emails critiques: {len(critical)} détectés")
            for critical_email in critical[:2]:
                subject = self.truncate_text(critical_email.get('subject', ''), 70)
                reasons = self.format_flags(critical_email.get('flags', []))
                score = critical_email.get('criticality_score', 0)
                lines.append(f"      - {subject} (score: {score}, causes: {reasons})")
        return lines

    def connect(self) -> bool:
        """Connexion au serveur IMAP"""
        try:
            logging.info(f"Tentative de connexion à {self.imap_server}:{self.port}")
            try:
                self.mail = imaplib.IMAP4_SSL(self.imap_server, self.port)
                logging.info("Connexion SSL établie")
            except Exception as ssl_error:
                logging.warning(f"Échec SSL: {ssl_error}")
                logging.info("Tentative de connexion sans SSL...")
                self.mail = imaplib.IMAP4(self.imap_server, 143)
                logging.info("Connexion non-SSL établie")
            
            logging.info(f"Tentative d'authentification pour {self.email_address}")
            self.mail.login(self.email_address, self.password)
            logging.info("Connexion réussie")
            return True
        except Exception as e:
            logging.error(f"Erreur de connexion: {e}")
            return False

    def disconnect(self):
        """Déconnexion du serveur IMAP"""
        if self.mail:
            try:
                self.mail.close()
                self.mail.logout()
                logging.info("Déconnexion réussie")
            except:
                pass

    def decode_header_value(self, value: str) -> str:
        """Décode les en-têtes d'email"""
        if not value:
            return ""
        decoded_parts = decode_header(value)
        decoded_string = ""
        for part, encoding in decoded_parts:
            try:
                if isinstance(part, bytes):
                    decoded_string += part.decode(encoding or 'utf-8', errors='ignore')
                else:
                    decoded_string += part
            except:
                pass
        return decoded_string

    def extract_email_content(self, msg) -> Dict:
        """Extrait le contenu d'un email"""
        content = {
            'subject': self.decode_header_value(msg['Subject']),
            'from': self.decode_header_value(msg['From']),
            'to': self.decode_header_value(msg['To']),
            'date': msg['Date'],
            'body': ""
        }

        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        if body:
                            content['body'] = self.decode_best_effort(
                                body, part.get_content_charset()
                            )
                            break
            else:
                body = msg.get_payload(decode=True)
                if body:
                    content['body'] = self.decode_best_effort(
                        body, msg.get_content_charset()
                    )
        except:
            pass

        content['normalized_text'] = self.normalize_email_text(content)
        return content

    def search_project_emails(self, project_filters: List[str], days_back: int = 30) -> Dict:
        """Recherche les emails concernant les projets spécifiés"""
        if not self.mail:
            return {}

        try:
            status, count = self.mail.select('INBOX')
            if status != 'OK':
                return {}

            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
            search_criteria = f'(SINCE "{start_date}")'
            
            status, messages = self.mail.search(None, search_criteria)
            if status != 'OK':
                return {}

            email_ids = messages[0].split()
            logging.info(f"Trouvé {len(email_ids)} emails à analyser")

            project_data = defaultdict(lambda: {
                'emails': [], 'participants': set(), 'keywords': defaultdict(int), 'dates': []
            })

            for i, email_id in enumerate(email_ids):
                try:
                    if i % 20 == 0:
                        logging.info(f"Traitement: {i+1}/{len(email_ids)} emails")

                    cache_key = email_id.decode(errors='ignore') if isinstance(email_id, bytes) else str(email_id)
                    cache_entry = self.email_cache.get(cache_key)
                    if cache_entry:
                        email_content = cache_entry.get('content', {})
                        matching_projects = cache_entry.get('projects', [])
                    else:
                        header_start = perf_counter()
                        status, header_data = self.mail.fetch(
                            email_id,
                            '(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM TO DATE)])'
                        )
                        self.record_timing("imap_fetch_headers", perf_counter() - header_start)
                        if status != 'OK' or not header_data or not header_data[0]:
                            continue

                        header_msg = email.message_from_bytes(header_data[0][1])
                        header_subject = self.decode_header_value(header_msg.get('Subject'))
                        if not self.should_fetch_full_message(header_subject, project_filters):
                            self.email_cache[cache_key] = {'projects': [], 'content': {}}
                            continue

                        fetch_start = perf_counter()
                        status, msg_data = self.mail.fetch(email_id, '(RFC822)')
                        self.record_timing("imap_fetch_full_message", perf_counter() - fetch_start)
                        if status != 'OK':
                            continue

                        parse_start = perf_counter()
                        msg = email.message_from_bytes(msg_data[0][1])
                        email_content = self.extract_email_content(msg)
                        self.record_timing("email_extract_content", perf_counter() - parse_start)

                        matching_projects = self.check_project_relevance(email_content, project_filters)
                        self.email_cache[cache_key] = {
                            'projects': matching_projects,
                            'content': email_content
                        }

                    if not matching_projects or not email_content:
                        continue

                    for project in matching_projects:
                        project_data[project]['emails'].append(email_content)
                        project_data[project]['dates'].append(email_content.get('date'))
                        self.extract_participants(email_content, project_data[project]['participants'])

                except Exception as e:
                    continue

            self.save_cache()
            return dict(project_data)

        except Exception as e:
            logging.error(f"Erreur recherche: {e}")
            return {}

    def check_project_relevance(self, email_content: Dict, project_filters: List[str]) -> List[str]:
        """Vérifie si un email concerne un projet spécifique"""
        matching_projects = []
        search_text = email_content.get('normalized_text')
        if not search_text:
            search_text = f"{email_content.get('subject', '')} {email_content.get('body', '')}".lower()
        for project_filter in project_filters:
            if project_filter.lower() in search_text:
                matching_projects.append(project_filter)
        return matching_projects

    def extract_participants(self, email_content: Dict, participants: set):
        """Extrait les participants du projet"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for field in ['from', 'to']:
            if email_content[field]:
                emails = re.findall(email_pattern, email_content[field])
                participants.update(emails)

    def generate_intelligent_summary(self, project_data: Dict) -> Dict:
        """Génère un résumé intelligent avec IA"""
        intelligent_summary = {}
        
        for project_name, data in project_data.items():
            logging.info(f"Analyse IA pour le projet: {project_name}")
            emails_all = data['emails']
            deep_emails = emails_all[-self.max_deep_emails:]
            
            # Analyses IA
            sentiment_start = perf_counter()
            sentiment_analysis = self.ai_analyzer.analyze_sentiment(deep_emails)
            self.record_timing("ia_sentiment", perf_counter() - sentiment_start)

            entities_start = perf_counter()
            entities = self.ai_analyzer.extract_entities(deep_emails)
            self.record_timing("ia_entities", perf_counter() - entities_start)

            summary_start = perf_counter()
            auto_summary = self.ai_analyzer.generate_auto_summary(deep_emails)
            self.record_timing("ia_summary", perf_counter() - summary_start)

            risk_start = perf_counter()
            risk_assessment = self.ai_analyzer.calculate_risk_score(emails_all, sentiment_analysis, entities)
            self.record_timing("ia_risk", perf_counter() - risk_start)

            critical_start = perf_counter()
            critical_emails = self.ai_analyzer.identify_critical_emails(emails_all)
            self.record_timing("ia_critical", perf_counter() - critical_start)
            
            # Données traditionnelles
            participants_list = list(data['participants'])
            email_count = len(emails_all)
            
            intelligent_summary[project_name] = {
                # Métriques de base
                'nb_emails': email_count,
                'nb_participants': len(participants_list),
                'participants': participants_list[:10],
                'emails_analyzes_en_profondeur': len(deep_emails),
                
                # Analyses IA
                'analyse_sentiment': sentiment_analysis,
                'entités_extraites': entities,
                'résumé_automatique': auto_summary,
                'évaluation_risque': risk_assessment,
                'emails_critiques': critical_emails,
                
                # Métriques combinées
                'score_activité': email_count + len(entities.get('technologies', [])),
                'priorité_attention': 'HAUTE' if risk_assessment.get('niveau_risque') == 'CRITIQUE' else 'NORMALE'
            }
        
        return intelligent_summary

def main():
    parser = argparse.ArgumentParser(description='Analyse intelligente des emails de projets avec IA')
    parser.add_argument('--email', required=True, help='Adresse email')
    parser.add_argument('--password', required=True, help='Mot de passe')
    parser.add_argument('--projects', required=True, nargs='+', help='Liste des projets à analyser')
    parser.add_argument('--days', type=int, default=30, help='Nombre de jours à analyser')
    parser.add_argument('--server', default='mail.mediasoftci.net', help='Serveur IMAP')
    parser.add_argument('--port', type=int, default=993, help='Port IMAP')
    parser.add_argument('--no-ssl', action='store_true', help='Désactiver SSL')
    parser.add_argument('--output', help='Fichier de sortie JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux')
    parser.add_argument(
        '--max-deep-emails',
        type=int,
        default=20,
        help="Nombre max d'emails analysés en profondeur par projet"
    )
    parser.add_argument(
        '--cache-file',
        default='email_analysis_cache.json',
        help='Fichier de cache local pour éviter les retraitements'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    analyzer = EmailProjectAnalyzer(
        args.email,
        args.password,
        args.server,
        args.port,
        max_deep_emails=args.max_deep_emails,
        cache_file=args.cache_file
    )

    try:
        print("🤖 Démarrage de l'analyse intelligente des emails...")
        
        if not analyzer.connect():
            print("❌ Échec de la connexion")
            return

        search_start = perf_counter()
        project_data = analyzer.search_project_emails(args.projects, args.days)
        analyzer.record_timing("phase_recherche_imap", perf_counter() - search_start)
        
        if not project_data:
            print("❌ Aucun email trouvé")
            return

        print("🧠 Analyse IA en cours...")
        ai_start = perf_counter()
        intelligent_summary = analyzer.generate_intelligent_summary(project_data)
        analyzer.record_timing("phase_analyse_ia", perf_counter() - ai_start)

        # Affichage des résultats
        print("\n" + "="*80)
        print("🤖 RAPPORT INTELLIGENT D'ANALYSE DES PROJETS")
        print("="*80)

        for project_name, analysis in intelligent_summary.items():
            for line in analyzer.format_project_report(project_name, analysis):
                print(line)
        
        # Sauvegarde
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(intelligent_summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Rapport sauvegardé: {args.output}")

        print("\n⏱️ BILAN PERFORMANCE")
        print("-" * 80)
        for step in analyzer.get_top_timing_steps(3):
            print(
                f"   • {step['étape']}: {step['total_s']}s "
                f"(runs: {step['runs']}, moyenne: {step['moyenne_s']}s)"
            )

        print(f"\n✅ Analyse intelligente terminée!")

    except Exception as e:
        logging.error(f"Erreur: {e}")
    finally:
        analyzer.disconnect()

if __name__ == "__main__":
    main()