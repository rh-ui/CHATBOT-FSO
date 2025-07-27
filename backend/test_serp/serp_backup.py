from playwright.sync_api import sync_playwright
import time
import random

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
                '--disable-ipc-flooding-protection'
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
        print(f"🔍 Recherche sur : {search_url}")
        page.goto(search_url)

        # Handle consent popup if present (multiple variations)
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
        
        try:
            for selector in consent_selectors:
                try:
                    page.wait_for_selector(selector, timeout=2000)
                    consent_button = page.locator(selector)
                    if consent_button.is_visible():
                        consent_button.click()
                        page.wait_for_load_state('networkidle')
                        print("✅ Consentement accepté")
                        break
                except:
                    continue
        except Exception as e:
            print(f"⚠️ Erreur gestion consentement: {e}")

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
                page.wait_for_selector(indicator, timeout=1000)
                captcha_found = True
                print("🤖 CAPTCHA détecté ! Veuillez le résoudre manuellement...")
                print("En attente de résolution du CAPTCHA...")
                
                # Wait for user to solve CAPTCHA (check if search results appear)
                for i in range(60):  # Wait up to 60 seconds
                    try:
                        if page.locator('div#search, div#rso').is_visible():
                            print("✅ CAPTCHA résolu ! On continue...")
                            break
                    except:
                        pass
                    time.sleep(1)
                    if i == 59:
                        print("❌ Timeout attente résolution CAPTCHA")
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
            print("div#search non trouvé, tentative avec div#rso")
            try:
                page.wait_for_selector('div#rso', timeout=5000)
            except Exception:
                print("Aucun résultat trouvé, lecture du contenu complet de la page...")

        # Optional extra wait for page to stabilize
        time.sleep(2)

        # Get search result snippets
        snippets = page.locator('div#search').all_inner_texts()
        if not snippets:
            snippets = page.locator('div#rso').all_inner_texts()
        
        # If still no results, try getting all text from body
        if not snippets:
            print("Tentative récupération de tout le contenu de la page...")
            snippets = [page.locator('body').inner_text()]

        print("=== Résultats ===")
        found = False
        for text in snippets:
            print(text)
            if keyword.lower() in text.lower():
                found = True

        if found:
            print(f"\n✅ Mot-clé '{keyword}' trouvé dans les résultats !")
        else:
            print(f"\n❌ Mot-clé '{keyword}' NON trouvé dans les résultats.")

        browser.close()


if __name__ == "__main__":
    search_query = "date de nomination du doyen de la FSO?"
    google_search_and_check(search_query)
