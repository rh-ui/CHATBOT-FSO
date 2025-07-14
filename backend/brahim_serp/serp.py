from playwright.sync_api import sync_playwright
import time
import random


prompt = """
Rules:
if question is not related to FSO, you should not answer."
if question is related to FSO, you should answer it in a concise and accurate manner.
Déclencher une recherche web
– Si la question concerne une info récente ou administrative (ex: « Qui est le doyen actuel ? »), faire une recherche web automatique.
– Exemple technique en Python : requests + BeautifulSoup pour récupérer les snippets Google.

Filtrer les sources fiables
– Ne garder que les résultats venant de domaines officiels ou universitaires (par ex: ump.ma, fso.ump.ma, .gov.ma).
– Utiliser une whitelist ou vérifier que le domaine est dans la liste.

Extraire la réponse
– Ouvrir la page et extraire le texte principal.
– Chercher la réponse précise (nom, date, titre) avec regex ou en scannant le texte.

Vérifier sémantiquement (optionnel mais conseillé)
– Utiliser un modèle NLP (spaCy ou HuggingFace) pour détecter et valider que la réponse est bien une personne, une fonction, ou une date.
use :
| But                 | Outil/API                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| Recherche web       | [SerpAPI](https://serpapi.com/), [Bing Search API](https://learn.microsoft.com/en-us/bing/search-apis/) |
| Scraping HTML       | `requests` + `BeautifulSoup`, `playwright` (pour JS)                                                    |
| Traitement du texte | `spaCy`, `transformers` (HuggingFace)                                                                   |
| Récupération de PDF | `PyMuPDF`, `pdfplumber`, `unstructured`                                                                 |
"""

def google_search_and_check(query, keyword="Maarouf"):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--disable-default-apps',
                '--disable-features=TranslateUI',
                '--disable-ipc-flooding-protectio n'
            ]
        )
        
        # Create context with realistic user agent
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()

        # Add some randomness to mimic human behavior
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)

        search_url = f"https://www.google.com/search?q={query}"
        page.goto(search_url)

        # Handle consent popup if present (multiple variations)
        try:
            # French consent
            consent_selectors = [
                'button:has-text("J\'accepte")',
                'button:has-text("Tout accepter")',
                'button:has-text("Accepter tout")',
                'button[id*="accept"]',
                'button[aria-label*="Accept"]',
                'button:has-text("Accept all")',
                'button:has-text("I agree")',
                'div[role="button"]:has-text("Accept")',
                '#L2AGLb'  # Common Google consent button ID
            ]
            
            for selector in consent_selectors:
                try:
                    consent_button = page.locator(selector)
                    if consent_button.is_visible(timeout=2000):
                        consent_button.click()
                        page.wait_for_load_state('networkidle')
                        break
                except:
                    continue
        except Exception:
            pass

        # Wait a bit and add random mouse movement
        time.sleep(random.uniform(2, 4))
        
        # Check for CAPTCHA and handle it
        captcha_indicators = [
            'div:has-text("unusual traffic")',
            'div:has-text("trafic inhabituel")',
            'div:has-text("robot")',
            'div:has-text("automated")',
            'div:has-text("CAPTCHA")',
            'iframe[src*="recaptcha"]',
            'div[id*="captcha"]',
            'div.g-recaptcha'
        ]
        
        captcha_found = False
        for indicator in captcha_indicators:
            try:
                if page.locator(indicator).is_visible(timeout=1000):
                    captcha_found = True
                    print("🤖 CAPTCHA detected! Please solve it manually...")
                    print("Waiting for you to solve the CAPTCHA...")
                    
                    # Wait for user to solve CAPTCHA (check if search results appear)
                    for i in range(60):  # Wait up to 60 seconds
                        try:
                            if page.locator('div#search, div#rso').is_visible(timeout=1000):
                                print("✅ CAPTCHA solved! Continuing...")
                                break
                        except:
                            pass
                        time.sleep(1)
                        if i == 59:
                            print("❌ Timeout waiting for CAPTCHA solution")
                            browser.close()
                            return
                    break
            except:
                continue

        if not captcha_found:
            # Add some human-like behavior
            page.mouse.move(random.randint(100, 500), random.randint(100, 300))
            time.sleep(random.uniform(1, 2))

        # Wait for search results or fallback to body
        try:
            page.wait_for_selector('div#search', timeout=10000)
        except Exception:
            print("div#search not found, trying div#rso")
            try:
                page.wait_for_selector('div#rso', timeout=5000)
            except Exception:
                print("No search results found, checking page content...")

        # Optional extra wait for page to stabilize
        time.sleep(2)

        # Try to click on business hours to expand them
        try:
            # Look for "Fermé" or hours indicators and click to expand
            hours_selectors = [
                'span:has-text("Fermé")',
                'span:has-text("Ouvert")',
                'span:has-text("Ferme")',
                'div[data-attrid="kc:/location/location:hours"]',
                'div[jsname="GYGCjb"]',
                '[aria-label*="horaires"]',
                '[aria-label*="hours"]'
            ]
            
            for selector in hours_selectors:
                try:
                    hours_element = page.locator(selector).first
                    if hours_element.is_visible(timeout=2000):
                        print(f"Found hours element with selector: {selector}")
                        hours_element.click()
                        time.sleep(2)  # Wait for expansion
                        break
                except:
                    continue
        except Exception as e:
            print(f"Could not click on hours: {e}")

        # Look for business information panel
        business_info = ""
        try:
            # Try to get business panel information
            business_selectors = [
                'div[data-attrid="kc:/location/location:address"]',
                'div[data-attrid="kc:/location/location:phone"]',
                'div[data-attrid="kc:/location/location:hours"]',
                'div[jsname="GYGCjb"]',
                'div[role="main"] div[data-ved]',
                'div[data-hveid] div[data-ved]'
            ]
            
            for selector in business_selectors:
                try:
                    elements = page.locator(selector)
                    if elements.count() > 0:
                        for i in range(elements.count()):
                            text = elements.nth(i).inner_text()
                            if text and len(text.strip()) > 0:
                                business_info += f"{text}\n"
                except:
                    continue
        except Exception as e:
            print(f"Error getting business info: {e}")

        # Get detailed schedule if available
        schedule_info = ""
        try:
            # Look for expanded schedule
            schedule_selectors = [
                'div[data-attrid="kc:/location/location:hours"] table',
                'div[data-attrid="kc:/location/location:hours"] div',
                'table[aria-label*="horaires"]',
                'table[aria-label*="hours"]',
                'div[jsname="GYGCjb"] table',
                'div[jsname="GYGCjb"] div'
            ]
            
            for selector in schedule_selectors:
                try:
                    schedule_elements = page.locator(selector)
                    if schedule_elements.count() > 0:
                        for i in range(schedule_elements.count()):
                            text = schedule_elements.nth(i).inner_text()
                            if text and len(text.strip()) > 0:
                                schedule_info += f"{text}\n"
                except:
                    continue
        except Exception as e:
            print(f"Error getting schedule info: {e}")

        # Get regular search results
        snippets = page.locator('div#search').all_inner_texts()
        if not snippets:
            snippets = page.locator('div#rso').all_inner_texts()
        
        # If still no results, try getting all text from body
        if not snippets:
            print("Trying to get all page content...")
            snippets = [page.locator('body').inner_text()]

        print("=== Business Information ===")
        if business_info:
            print(business_info)
        
        print("=== Schedule Information ===")
        if schedule_info:
            print(schedule_info)
        
        print("=== Search Results ===")
        found = False
        for text in snippets:
            print(text)
            if keyword.lower() in text.lower():
                found = True

        # Check if keyword is found in business info or schedule too
        if not found and business_info and keyword.lower() in business_info.lower():
            found = True
        if not found and schedule_info and keyword.lower() in schedule_info.lower():
            found = True

        if found:
            print(f"\n✅ Found keyword: '{keyword}' in results!")
        else:
            print(f"\n❌ Keyword '{keyword}' NOT found in results.")

        browser.close()

if __name__ == "__main__":
    search_query = "date de nomination du doyen de la FSO?"
    google_search_and_check(search_query)