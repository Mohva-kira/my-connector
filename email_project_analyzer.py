"""
Email Project Analyzer - Version Corrigée et Complète
Analyse des emails liés à des projets, extraction participants, mots-clés et génération de résumé.
Gère correctement les datetime naïves/aware et les encodages HTML.
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

# Modules externes
from bs4 import BeautifulSoup
import chardet

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------ Classe principale ------------------
class EmailProjectAnalyzer:
    def __init__(self, email_address: str, password: str, imap_server: str = "mail.mediasoftci.net", port: int = 993):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.port = port
        self.mail = None
        self.project_emails = defaultdict(list)

    # ------------------ Connexion ------------------
    def connect(self) -> bool:
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
        except imaplib.IMAP4.error as imap_error:
            logging.error(f"Erreur IMAP: {imap_error}")
            return False
        except Exception as e:
            logging.error(f"Erreur de connexion: {e}")
            return False

    def disconnect(self):
        if self.mail:
            try:
                self.mail.close()
                self.mail.logout()
                logging.info("Déconnexion réussie")
            except Exception as e:
                logging.warning(f"Erreur lors de la déconnexion: {e}")

    # ------------------ Décodage Header ------------------
    def decode_header_value(self, value: str) -> str:
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
            except Exception as e:
                logging.warning(f"Erreur décodage header '{value}': {e}")
        return decoded_string

    # ------------------ Extraction du corps ------------------
    def extract_email_content(self, msg) -> Dict:
        content = {
            'subject': self.decode_header_value(msg['Subject']),
            'from': self.decode_header_value(msg['From']),
            'to': self.decode_header_value(msg['To']),
            'date': msg['Date'],
            'body': ""
        }

        body = ""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == "text/plain":
                        body_bytes = part.get_payload(decode=True)
                        if body_bytes:
                            detected = chardet.detect(body_bytes)
                            body = body_bytes.decode(detected['encoding'] or 'utf-8', errors='ignore')
                            break
                    elif ctype == "text/html" and not body:
                        body_bytes = part.get_payload(decode=True)
                        if body_bytes:
                            detected = chardet.detect(body_bytes)
                            html = body_bytes.decode(detected['encoding'] or 'utf-8', errors='ignore')
                            body = BeautifulSoup(html, "html.parser").get_text(separator="\n")
            else:
                body_bytes = msg.get_payload(decode=True)
                if body_bytes:
                    detected = chardet.detect(body_bytes)
                    body = body_bytes.decode(detected['encoding'] or 'utf-8', errors='ignore')
        except Exception as e:
            logging.warning(f"Erreur extraction body: {e}")

        content['body'] = body
        return content

    # ------------------ Recherche emails projets ------------------
    def search_project_emails(self, project_filters: List[str], days_back: int = 30) -> Dict:
        if not self.mail:
            logging.error("Pas de connexion active")
            return {}

        try:
            status, count = self.mail.select('INBOX')
            if status != 'OK':
                logging.error("Impossible de sélectionner la boîte de réception")
                return {}

            logging.info(f"Boîte de réception sélectionnée - {count[0].decode()} emails")

            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
            search_criteria = f'(SINCE "{start_date}")'
            logging.info(f"Critères de recherche: {search_criteria}")

            status, messages = self.mail.uid('SEARCH', None, search_criteria)
            if status != 'OK':
                logging.error("Erreur lors de la recherche d'emails")
                return {}

            email_uids = messages[0].split()
            logging.info(f"Trouvé {len(email_uids)} emails à analyser dans les {days_back} derniers jours")
            if not email_uids:
                logging.info("Aucun email trouvé dans la période spécifiée")
                return {}

            project_data = defaultdict(lambda: {
                'emails': [], 'participants': set(), 'keywords': defaultdict(int), 'dates': []
            })

            for i, uid in enumerate(email_uids, 1):
                try:
                    if i % 20 == 0:
                        logging.info(f"Traitement: {i}/{len(email_uids)} emails")
                    status, msg_data = self.mail.uid('FETCH', uid, '(RFC822)')
                    if status != 'OK':
                        continue

                    msg = email.message_from_bytes(msg_data[0][1])
                    email_content = self.extract_email_content(msg)

                    matching_projects = self.check_project_relevance(email_content, project_filters)
                    if matching_projects:
                        for project in matching_projects:
                            project_data[project]['emails'].append(email_content)
                            project_data[project]['dates'].append(email_content['date'])
                            self.extract_participants(email_content, project_data[project]['participants'])
                            self.extract_keywords(email_content, project_data[project]['keywords'])

                except Exception as e:
                    logging.warning(f"Erreur traitement email UID {uid}: {e}")
                    continue

            logging.info(f"Analyse terminée")
            return dict(project_data)

        except Exception as e:
            logging.error(f"Erreur lors de la recherche: {e}")
            return {}

    # ------------------ Vérification pertinence ------------------
    def check_project_relevance(self, email_content: Dict, project_filters: List[str]) -> List[str]:
        matching_projects = []
        search_text = f"{email_content['subject']} {email_content['body']}".lower()
        for project_filter in project_filters:
            if project_filter.lower() in search_text:
                matching_projects.append(project_filter)
        return matching_projects

    # ------------------ Extraction participants ------------------
    def extract_participants(self, email_content: Dict, participants: set):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for field in ['from', 'to']:
            if email_content[field]:
                emails = re.findall(email_pattern, email_content[field])
                participants.update(emails)

    # ------------------ Extraction mots-clés ------------------
    def extract_keywords(self, email_content: Dict, keywords: defaultdict):
        project_keywords = [
            'deadline','échéance','livraison','milestone','étape',
            'budget','coût','prix','devis','facture',
            'réunion','meeting','call','appel','rdv',
            'urgent','priorité','important','critique',
            'terminé','fini','complété','livré','deployed',
            'projet','project','développement','development',
            'test','bug','issue','problème','solution'
        ]
        text = f"{email_content['subject']} {email_content['body']}".lower()
        for keyword in project_keywords:
            if keyword in text:
                keywords[keyword] += text.count(keyword)

    # ------------------ Génération résumé ------------------
    def generate_summary(self, project_data: Dict) -> Dict:
        summary = {}
        for project_name, data in project_data.items():
            participants_list = list(data['participants'])
            top_keywords = dict(sorted(data['keywords'].items(), key=lambda x: x[1], reverse=True)[:10])
            email_count = len(data['emails'])
            participant_count = len(participants_list)

            # Gestion des dates safe
            dates = [d for d in data['dates'] if d]
            date_range = ""
            if dates:
                parsed_dates = []
                for d in dates:
                    try:
                        if '(' in d:
                            d = d.split(' (')[0]
                        dt = email.utils.parsedate_to_datetime(d)
                        if dt.tzinfo is not None:
                            dt = dt.astimezone(tz=None).replace(tzinfo=None)
                        parsed_dates.append(dt)
                    except Exception as e:
                        logging.warning(f"Erreur parsing date '{d}': {e}")
                        continue
                if parsed_dates:
                    min_date = min(parsed_dates).strftime('%d/%m/%Y')
                    max_date = max(parsed_dates).strftime('%d/%m/%Y')
                    date_range = f"{min_date} - {max_date}" if min_date != max_date else min_date

            recent_subjects = [e['subject'] for e in data['emails'][-3:] if e['subject']]

            summary[project_name] = {
                'nb_emails': email_count,
                'nb_participants': participant_count,
                'participants': participants_list,
                'periode': date_range,
                'mots_cles_principaux': top_keywords,
                'sujets_recents': recent_subjects,
                'activite_recente': email_count > 0,
                'score_activite': email_count + len(top_keywords)
            }

        return summary

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser(description='Analyse des emails de projets')
    parser.add_argument('--email', required=False, help='Adresse email')
    parser.add_argument('--password', required=False, help='Mot de passe')
    parser.add_argument('--projects', required=True, nargs='+', help='Liste des projets à analyser')
    parser.add_argument('--days', type=int, default=30, help='Nombre de jours à analyser (défaut: 30)')
    parser.add_argument('--server', default='mail.mediasoftci.net', help='Serveur IMAP')
    parser.add_argument('--port', type=int, default=993, help='Port IMAP')
    parser.add_argument('--no-ssl', action='store_true', help='Désactiver SSL')
    parser.add_argument('--output', help='Fichier de sortie JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    email_addr = args.email or os.getenv("PROJECT_EMAIL")
    password = args.password or os.getenv("PROJECT_EMAIL_PASSWORD")
    if not email_addr or not password:
        logging.error("Email et mot de passe requis. CLI ou variables d'environnement.")
        return

    if args.no_ssl and args.port == 993:
        args.port = 143

    analyzer = EmailProjectAnalyzer(email_addr, password, args.server, args.port)

    try:
        print("🔍 Démarrage de l'analyse des emails de projets...")
        if not analyzer.connect():
            logging.error("Impossible de se connecter. Vérifiez vos identifiants et le serveur.")
            return

        project_data = analyzer.search_project_emails(args.projects, args.days)
        if not project_data:
            print(f"\n❌ Aucun email trouvé pour les projets spécifiés dans les {args.days} derniers jours")
            return

        summary = analyzer.generate_summary(project_data)

        print("\n📊 RÉSUMÉ DES PROJETS ANALYSÉS\n" + "="*70)
        sorted_projects = sorted(summary.items(), key=lambda x: x[1]['score_activite'], reverse=True)
        for project_name, stats in sorted_projects:
            print(f"\n🚀 PROJET: {project_name}")
            print(f"   📧 Emails analysés: {stats['nb_emails']}")
            print(f"   👥 Participants: {stats['nb_participants']}")
            print(f"   📅 Période: {stats['periode']}")
            print(f"   ⚡ Activité récente: {'✅ Oui' if stats['activite_recente'] else '❌ Non'}")
            print(f"   📊 Score d'activité: {stats['score_activite']}")
            if stats['participants']:
                print(f"   👨‍💼 Équipe: {', '.join(stats['participants'][:5])}")
                if len(stats['participants']) > 5:
                    print(f"      ... et {len(stats['participants']) - 5} autres")
            if stats['mots_cles_principaux']:
                print("   🔑 Mots-clés principaux:")
                for k, v in list(stats['mots_cles_principaux'].items())[:5]:
                    print(f"      - {k}: {v} mentions")
            if stats['sujets_recents']:
                print("   📋 Sujets récents:")
                for subject in stats['sujets_recents']:
                    print(f"      - {subject[:60]}{'...' if len(subject) > 60 else ''}")

        total_emails = sum(stats['nb_emails'] for stats in summary.values())
        total_participants = len(set().union(*[set(stats['participants']) for stats in summary.values()]))

        print("\n📈 STATISTIQUES GLOBALES\n" + "="*70)
        print(f"   📊 Total projets analysés: {len(summary)}")
        print(f"   📧 Total emails traités: {total_emails}")
        print(f"   👥 Total participants uniques: {total_participants}")
        print(f"   📅 Période d'analyse: {args.days} jours")

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Résumé sauvegardé dans: {args.output}")

        print("\n✅ Analyse terminée avec succès!")

    except KeyboardInterrupt:
        print("\n⏹️ Analyse interrompue par l'utilisateur")
    except Exception as e:
        logging.error(f"Erreur durant l'analyse: {e}")
        print(f"\n❌ Erreur durant l'analyse: {e}")
    finally:
        analyzer.disconnect()

if __name__ == "__main__":
    main()
