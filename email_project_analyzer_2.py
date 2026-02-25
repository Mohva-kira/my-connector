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
                text = f"{email_data['subject']} {email_data['body']}"[:512]
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
                text = f"{email_data['subject']} {email_data['body']}"
                
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
    def __init__(self, email_address: str, password: str, imap_server: str = "mail.mediasoftci.net", port: int = 993):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.port = port
        self.mail = None
        self.project_emails = defaultdict(list)
        
        # Initialiser l'analyseur IA
        self.ai_analyzer = EmailIntelligentAnalyzer()

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
                            content['body'] = body.decode('utf-8', errors='ignore')
                            break
            else:
                body = msg.get_payload(decode=True)
                if body:
                    content['body'] = body.decode('utf-8', errors='ignore')
        except:
            pass

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
                    
                    status, msg_data = self.mail.fetch(email_id, '(RFC822)')
                    if status != 'OK':
                        continue

                    msg = email.message_from_bytes(msg_data[0][1])
                    email_content = self.extract_email_content(msg)

                    matching_projects = self.check_project_relevance(email_content, project_filters)
                    for project in matching_projects:
                        project_data[project]['emails'].append(email_content)
                        project_data[project]['dates'].append(email_content['date'])
                        self.extract_participants(email_content, project_data[project]['participants'])

                except Exception as e:
                    continue

            return dict(project_data)

        except Exception as e:
            logging.error(f"Erreur recherche: {e}")
            return {}

    def check_project_relevance(self, email_content: Dict, project_filters: List[str]) -> List[str]:
        """Vérifie si un email concerne un projet spécifique"""
        matching_projects = []
        search_text = f"{email_content['subject']} {email_content['body']}".lower()
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
            
            # Analyses IA
            sentiment_analysis = self.ai_analyzer.analyze_sentiment(data['emails'])
            entities = self.ai_analyzer.extract_entities(data['emails'])
            auto_summary = self.ai_analyzer.generate_auto_summary(data['emails'])
            risk_assessment = self.ai_analyzer.calculate_risk_score(data['emails'], sentiment_analysis, entities)
            critical_emails = self.ai_analyzer.identify_critical_emails(data['emails'])
            
            # Données traditionnelles
            participants_list = list(data['participants'])
            email_count = len(data['emails'])
            
            intelligent_summary[project_name] = {
                # Métriques de base
                'nb_emails': email_count,
                'nb_participants': len(participants_list),
                'participants': participants_list[:10],
                
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

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    analyzer = EmailProjectAnalyzer(args.email, args.password, args.server, args.port)

    try:
        print("🤖 Démarrage de l'analyse intelligente des emails...")
        
        if not analyzer.connect():
            print("❌ Échec de la connexion")
            return

        project_data = analyzer.search_project_emails(args.projects, args.days)
        
        if not project_data:
            print("❌ Aucun email trouvé")
            return

        print("🧠 Analyse IA en cours...")
        intelligent_summary = analyzer.generate_intelligent_summary(project_data)

        # Affichage des résultats
        print("\n" + "="*80)
        print("🤖 RAPPORT INTELLIGENT D'ANALYSE DES PROJETS")
        print("="*80)

        for project_name, analysis in intelligent_summary.items():
            print(f"\n🚀 PROJET: {project_name}")
            print(f"   📊 Priorité: {analysis['priorité_attention']}")
            print(f"   📧 Emails: {analysis['nb_emails']}")
            print(f"   👥 Participants: {analysis['nb_participants']}")
            
            # Analyse de sentiment
            sentiment = analysis['analyse_sentiment']
            if 'tendance' in sentiment:
                method = sentiment.get('méthode', 'N/A')
                print(f"   😊 Sentiment: {sentiment['tendance']} (Méthode: {method})")
            
            # Évaluation des risques
            risk = analysis['évaluation_risque']
            print(f"   ⚠️ Risque: {risk.get('niveau_risque', 'N/A')} (Score: {risk.get('score_risque', 0)})")
            print(f"   💡 {risk.get('recommandation', 'Aucune recommandation')}")
            
            # Entités importantes
            entities = analysis['entités_extraites']
            if entities.get('technologies'):
                print(f"   💻 Technologies: {', '.join(entities['technologies'][:5])}")
            if entities.get('montants'):
                print(f"   💰 Montants: {', '.join(entities['montants'][:3])}")
            
            # Résumé automatique
            auto_summary = analysis['résumé_automatique']
            if isinstance(auto_summary, dict) and 'résumé_automatique' in auto_summary:
                method = auto_summary.get('méthode', 'N/A')
                print(f"   📝 Résumé ({method}): {auto_summary['résumé_automatique'][:200]}...")
            
            # Emails critiques
            critical = analysis['emails_critiques']
            if critical:
                print(f"   🚨 Emails critiques: {len(critical)} détectés")
                for critical_email in critical[:2]:
                    print(f"      - {critical_email['subject'][:50]}... (Score: {critical_email['criticality_score']})")
        
        # Sauvegarde
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(intelligent_summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Rapport sauvegardé: {args.output}")

        print(f"\n✅ Analyse intelligente terminée!")

    except Exception as e:
        logging.error(f"Erreur: {e}")
    finally:
        analyzer.disconnect()

if __name__ == "__main__":
    main()