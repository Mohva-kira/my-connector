"""
Script de test pour vérifier la connexion email
"""
import imaplib
import logging
import getpass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_imap_connection():
    """Test de connexion IMAP avec différentes configurations"""
    
    email_address = input("Adresse email: ")
    password = getpass.getpass("Mot de passe: ")
    
    # Configurations communes
    configurations = [
        # Configuration pour mediasoftci.net
        {"server": "mail.mediasoftci.net", "port": 993, "ssl": True},
        {"server": "mail.mediasoftci.net", "port": 143, "ssl": False},
        {"server": "imap.mediasoftci.net", "port": 993, "ssl": True},
        {"server": "mediasoftci.net", "port": 993, "ssl": True},
        
        # Configurations génériques
        {"server": "outlook.office365.com", "port": 993, "ssl": True},
        {"server": "imap.gmail.com", "port": 993, "ssl": True},
    ]
    
    for config in configurations:
        try:
            logging.info(f"Test: {config['server']}:{config['port']} (SSL: {config['ssl']})")
            
            if config['ssl']:
                mail = imaplib.IMAP4_SSL(config['server'], config['port'])
            else:
                mail = imaplib.IMAP4(config['server'], config['port'])
            
            # Test de login
            mail.login(email_address, password)
            
            # Test de sélection de boîte
            mail.select('inbox')
            
            logging.info(f"✅ SUCCÈS avec {config['server']}:{config['port']}")
            
            # Affichage des dossiers disponibles
            status, folders = mail.list()
            if status == 'OK':
                print("Dossiers disponibles:")
                for folder in folders[:5]:  # Affiche les 5 premiers
                    print(f"  - {folder.decode()}")
            
            mail.logout()
            return config
            
        except Exception as e:
            logging.error(f"❌ Échec avec {config['server']}:{config['port']} - {e}")
            continue
    
    logging.error("Aucune configuration fonctionnelle trouvée")
    return None

if __name__ == "__main__":
    working_config = test_imap_connection()
    if working_config:
        print(f"\n🎉 Configuration recommandée:")
        print(f"Serveur: {working_config['server']}")
        print(f"Port: {working_config['port']}")
        print(f"SSL: {working_config['ssl']}")