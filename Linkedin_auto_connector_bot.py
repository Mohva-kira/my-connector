"""
This script is a LinkedIn bot that automatically sends connection requests with a custom note to profiles on LinkedIn.
It uses the Selenium WebDriver to navigate LinkedIn and interact with the UI elements.
Do 100 requests per week!!!!!
If not, LinkedIn will block your account.
Add my LinkedIn also - https://www.linkedin.com/in/mrbondarenko/
Replace your search link with keywords you need!
Go to LinkedIn main page, press on the search bar, put the keywords you need(Tech Recruter or Cloud Engineer for example),
press enter, select people only! copy the link and paste it in the SEARCH_LINK variable.
 Have fun!
"""

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, MoveTargetOutOfBoundsException, ElementClickInterceptedException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import logging
import time
from selenium.common.exceptions import StaleElementReferenceException, ElementClickInterceptedException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your LinkedIn credentials
LINKEDIN_USERNAME = 'mtandjo@gmail.com' # your email
LINKEDIN_PASSWORD = 'Flyentreprise2021' # your password

MAX_RETRIES = 5  # Maximum number of retries for refreshing

SEARCH_LINK = ("https://www.linkedin.com/search/results/people/?keywords=grh&origin=SWITCH_SEARCH_VERTICAL&sid=9CQ")
# Base connection message template
BASE_CONNECTION_MESSAGE = """Bonjour [first_name],

Ravi de découvrir votre profil.
Je me présente : Mohamed Tandjigora, ingénieur en informatique avec plus de 10 ans d’expérience dans la transformation digitale, la fintech et le développement de solutions métiers sur mesure.

J’accompagne les entreprises dans leur digitalisation en concevant des applications de gestion des recrutements 100% en ligne, inspirées des meilleures pratiques du marché (telles que celles d’Antares).
Ces solutions permettent de centraliser les candidatures, simplifier le suivi des entretiens, et accélérer le processus de sélection grâce à une interface intuitive pour les recruteurs et les candidats.

Je serais ravi d’échanger avec vous sur vos besoins en matière de digitalisation RH et de vous présenter un prototype adapté à votre activité.

Bien cordialement,
Mohamed Tandjigora
Ingénieur en informatique | Expert en transformation digitale
🌐 https://mtandjo.pro

📍 Basé à Abidjan – Intervention à l’international
"""

MAX_CONNECT_REQUESTS = 20  # Limit for connection requests

def login_to_linkedin(driver, username, password):
    try:
        driver.get("https://www.linkedin.com/login")
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "username")))

        # Enter username
        username_field = driver.find_element(By.ID, "username")
        username_field.send_keys(username)

        # Enter password
        password_field = driver.find_element(By.ID, "password")
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)
        time.sleep(5)  # Wait for the page to load or enter captcha
        WebDriverWait(driver, 20).until(EC.url_contains("/feed"))
        logging.info("Successfully logged into LinkedIn.")
        time.sleep(5)  # Wait for the feed to load
    except Exception as e:
        logging.error(f"Error during LinkedIn login: {e}")

def go_to_next_page(driver):
    try:
        time.sleep(5)  # Wait for the page to load
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # Scroll down
        next_page_button = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//button[@aria-label='Suivant']"))
        )
        next_page_button.click()
        logging.info("Naiguer sur la page suivante.")
        time.sleep(5)  # Wait for the new page to load
    except NoSuchElementException as e:
        logging.error(f"Element non trouvé: {e}")
        return False
    except Exception as e:
        logging.error(f"Error navigating to the next page: {e}")
        return False
    return True

def scrool_down(driver):
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # Scroll down
        time.sleep(5)  # Wait for the page to load
    except Exception as e:
        logging.error(f"Error during scrolling down: {e}")

def handle_connect_button_with_retry(driver, button_xpath):
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # Relocaliser le bouton pour éviter StaleElementReferenceException
            button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, button_xpath))
            )
            button.click()
            time.sleep(2)

            # Ajouter une note
            add_note_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By, "//button[@aria-label='Ajouter une note']"))
            )
            add_note_button.click()
            time.sleep(2)

            # Saisir le message personnalisé
            message_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//textarea[@name='message']"))
            )
            message_box.clear()
            message_box.send_keys(BASE_CONNECTION_MESSAGE)
            time.sleep(2)

            # Cliquer sur le bouton "Envoyer"
            send_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//span[contains(@class, 'artdeco-button__text') and text()='Envoyer']"))
            )
            send_button.click()
            logging.info("Sent connection request with a custom note.")
            time.sleep(2)
            return True  # Sortir de la fonction si la demande est envoyée avec succès

        except StaleElementReferenceException as e:
            logging.warning(f"StaleElementReferenceException: Relocalisation de l'élément... {e}")
            retry_count += 1
            time.sleep(2)  # Attendre avant de réessayer

        except ElementClickInterceptedException as e:
            logging.error(f"ElementClickInterceptedException: {e}")
            retry_count += 1
            time.sleep(2)  # Attendre avant de réessayer

        except TimeoutException as e:
            logging.error(f"TimeoutException: {e}")
            retry_count += 1
            time.sleep(2)  # Attendre avant de réessayer

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            retry_count += 1
            time.sleep(2)

    logging.error("Failed to send connection request after maximum retries.")
    return False
def handle_follow_button(button):
    try:
        button.click()
        logging.info("Followed the user.")
        time.sleep(1)
    except Exception as e:
        logging.error(f"Error handling 'Follow' button: {e}")

def process_buttons(driver):
    try:
        # Navigate to the search page
        driver.get(SEARCH_LINK)
        scrool_down(driver)
        time.sleep(5)

        connect_requests_sent = 0

        working = True


        while working:
            # Find all buttons on the page
            buttons = driver.find_elements(By.TAG_NAME, "button")

            # Count "Connect" and "Follow" buttons
            connect_buttons_count = sum(1 for button in buttons if button.text.strip().lower() == "se connecter")
            follow_buttons_count = sum(1 for button in buttons if button.text.strip().lower() == "suivre")
            logging.info(f"Total 'Connect' buttons on the page: {connect_buttons_count}")
            logging.info(f"Total 'Follow' buttons on the page: {follow_buttons_count}")

            # Process each "Connect" and "Follow" button
            for button in buttons:
                button_text = button.text.strip().lower()
                if button_text == "se connecter" and connect_requests_sent < MAX_CONNECT_REQUESTS:
                    handle_connect_button_with_retry(driver, button)
                    connect_requests_sent += 1
                    if connect_requests_sent >= MAX_CONNECT_REQUESTS:
                        logging.info(
                            f"Reached the limit of {MAX_CONNECT_REQUESTS} connection requests. Stopping connection requests.")
                        working = False
                        break
                    time.sleep(5)
                elif button_text == "suivre":
                    handle_follow_button(button)
                    time.sleep(5)

            # Attempt to navigate to the next page
            if not go_to_next_page(driver):
                logging.info("No more pages to process. Exiting.")
                break

            # Scroll down to load all elements on the new page
            scrool_down(driver)
            time.sleep(5)

    except Exception as e:
        logging.error(f"Error while processing buttons: {e}")


def refresh_page(driver, retries):
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Attempt {attempt}/{retries}: Refreshing the page.")
            driver.refresh()  # Refresh the page
            time.sleep(5)  # Wait for the page to reload
            return True
        except Exception as e:
            logging.error(f"Error during page refresh: {e}")

        if attempt == retries:
            logging.error("Maximum retries reached. Exiting the program.")
            driver.quit()
            exit(1)
    return False


if __name__ == "__main__":
    options = Options()
    options.binary_location = 'C:/Program Files/Mozilla Firefox/firefox.exe' ## path to your firefox browser(must install firefox browser)

    # Set up the webdriver (Replace the path with the path to your webdriver) // mine is geckodriver32.exe already installed in the directory
    # go to https://github.com/mozilla/geckodriver/releases to download latest version of geckodriver
    service = Service('geckodriver32.exe')
    driver = webdriver.Firefox(service=service, options=options)

    try:
        login_to_linkedin(driver, LINKEDIN_USERNAME, LINKEDIN_PASSWORD)
        process_buttons(driver)
    finally:
        driver.quit()
