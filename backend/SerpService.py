from playwright.sync_api import sync_playwright
import time
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
from LLMService import llm_service

# ✅ Liste blanche des domaines fiables
WHITELIST_DOMAINS = ["fso.ump.ma", ".gov.ma"]

def is_domain_allowed(url):
    """
    Vérifie si l'URL provient d'un domaine autorisé.
    """
    return any(domain in url for domain in WHITELIST_DOMAINS)

def extract_keywords(text):
    """
    Extrait les mots-clés importants d'un texte.
    """
    # Supprimer la ponctuation et convertir en minuscules
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned.split()
    
    
    stop_words = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 'donc', 'car', 
        'ni', 'or', 'ce', 'ces', 'cette', 'cet', 'se', 'sa', 'son', 'ses', 'leur', 'leurs',
        'que', 'qui', 'quoi', 'dont', 'où', 'par', 'pour', 'avec', 'sans', 'sous', 'sur',
        'dans', 'vers', 'chez', 'entre', 'depuis', 'pendant', 'avant', 'après', 'très', 
        'plus', 'moins', 'bien', 'mal', 'tout', 'tous', 'toute', 'toutes', 'même', 'autre'
    }
    
    return [word for word in words if len(word) > 2 and word not in stop_words]

def calculate_semantic_score(snippet, query):
    """
    Calcule un score sémantique sophistiqué basé sur plusieurs métriques.
    """
    scores = {}
    
    # 1. 📊 Score TF-IDF (similarité cosinus)
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([query, snippet])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        scores['tfidf'] = tfidf_score
    except:
        scores['tfidf'] = 0.0
    
    # 2. 🎯 Score de correspondance des mots-clés
    query_keywords = set(extract_keywords(query))
    snippet_keywords = set(extract_keywords(snippet))
    
    if query_keywords:
        keyword_match = len(query_keywords.intersection(snippet_keywords)) / len(query_keywords)
        scores['keyword_match'] = keyword_match
    else:
        scores['keyword_match'] = 0.0
    
    # 3. 📏 Score de densité des mots-clés
    snippet_words = extract_keywords(snippet)
    if snippet_words:
        keyword_density = sum(1 for word in snippet_words if word in query_keywords) / len(snippet_words)
        scores['keyword_density'] = keyword_density
    else:
        scores['keyword_density'] = 0.0
    
    # 4. 🎪 Score de proximité des mots-clés
    snippet_lower = snippet.lower()
    query_words = extract_keywords(query)
    proximity_score = 0.0
    
    if len(query_words) > 1:
        positions = []
        for word in query_words:
            pos = snippet_lower.find(word)
            if pos != -1:
                positions.append(pos)
        
        if len(positions) > 1:
            positions.sort()
            max_distance = max(positions) - min(positions)
            # Plus les mots sont proches, meilleur est le score
            proximity_score = 1.0 / (1.0 + max_distance / 100.0)
    
    scores['proximity'] = proximity_score
    
    # 5. 📖 Score de longueur optimale
    optimal_length = 150  # Longueur optimale d'un snippet
    length_penalty = abs(len(snippet) - optimal_length) / optimal_length
    length_score = max(0.0, 1.0 - length_penalty)
    scores['length'] = length_score
    
    # 🏆 Score final pondéré
    weights = {
        'tfidf': 0.35,
        'keyword_match': 0.25,
        'keyword_density': 0.15,
        'proximity': 0.15,
        'length': 0.10
    }
    
    final_score = sum(scores[metric] * weights[metric] for metric in scores)
    
    return final_score, scores

def filter_snippets_by_semantic_relevance(snippets, query, top_k=5, min_score=0.01):
    """
    Filtre et classe les snippets selon leur pertinence sémantique.
    """
    if not snippets:
        return []
    
    # Nettoyer les snippets
    cleaned_snippets = []
    for snippet in snippets:
        clean_snippet = snippet.strip()
        if len(clean_snippet) > 30:
            cleaned_snippets.append(clean_snippet)
    
    if not cleaned_snippets:
        return []
    
    # Calculer les scores pour chaque snippet
    scored_snippets = []
    for snippet in cleaned_snippets:
        final_score, detailed_scores = calculate_semantic_score(snippet, query)
        
        if final_score >= min_score:
            scored_snippets.append({
                'snippet': snippet,
                'final_score': final_score,
                'detailed_scores': detailed_scores
            })
    
    # Trier par score décroissant
    scored_snippets.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Retourner les top_k meilleurs
    return scored_snippets[:top_k]

def google_search_and_extract(query, lang, max_results=10):
    """
    Effectue une recherche Google et extrait les résultats avec scoring sémantique.
    """
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

        filtered_query = f"site:fso.ump.ma OR site:cg.gov.ma {query}"
        search_url = f"https://www.google.com/search?q={filtered_query}&num={max_results}&hl={lang}"

        try:
            print(f"🔍 Navigation vers: {search_url}")
            # page.goto(search_url)
            page.goto(search_url, wait_until="domcontentloaded", timeout=8000000)
            print(f"✅ Page chargée avec succès")
            
        except Exception as e:
            print(f"❌ Erreur de navigation: {e}")
            browser.close()
            return []

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
                    if consent_button.is_visible(timeout=200000):
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
                            return []
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

        # 🔍 DEBUG: Vérifier la présence d'éléments
        print("🔍 Recherche des sélecteurs possibles...")
        
        # Essayer différents sélecteurs pour les résultats Google
        possible_selectors = [
            "div.g",                           # Sélecteur classique
            "[data-sokoban-container] div",    # Nouveau format Google
            ".tF2Cxc",                        # Autre format possible
            ".g",                             # Version courte
            "div[data-ved]",                  # Basé sur l'attribut data-ved
        ]
        
        results = None
        working_selector = None
        
        for selector in possible_selectors:
            test_results = page.locator(selector)
            count = test_results.count()
            print(f"   🔍 {selector}: {count} éléments trouvés")
            
            if count > 0:
                results = test_results
                working_selector = selector
                break
        
        if not results or results.count() == 0:
            print("❌ Aucun résultat trouvé avec les sélecteurs disponibles")
            print("🔍 DEBUG: Structure de la page:")
            
            # Sauvegarder une capture d'écran pour debug
            try:
                page.screenshot(path="debug_google_results.png")
                print("📸 Capture d'écran sauvegardée: debug_google_results.png")
            except:
                pass
            
            # Essayer de trouver tous les liens
            all_links = page.locator("a[href*='http']")
            print(f"🔗 {all_links.count()} liens trouvés au total")
            
            browser.close()
            return []

        snippets = []
        links = []
        titles = []
        
        count = min(results.count(), max_results)
        print(f"📊 {count} résultats trouvés avec le sélecteur: {working_selector}")
        
        for i in range(count):
            try:
                result_element = results.nth(i)
                
                # Extraire le lien - essayer plusieurs méthodes
                link = ""
                link_selectors = ["a", "a[href*='http']", "[href*='http']"]
                for link_sel in link_selectors:
                    link_element = result_element.locator(link_sel).first
                    if link_element.count() > 0:
                        potential_link = link_element.get_attribute("href")
                        if potential_link and "http" in potential_link:
                            link = potential_link
                            break
                
                # Nettoyer les liens Google (supprimer les redirections)
                if link and "/url?q=" in link:
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                    if 'q' in parsed:
                        link = parsed['q'][0]
                
                # Extraire le titre
                title = ""
                title_selectors = ["h3", "[role='heading']", "h1", "h2"]
                for title_sel in title_selectors:
                    title_element = result_element.locator(title_sel).first
                    if title_element.count() > 0:
                        title = title_element.inner_text().strip()
                        break
                
                if not title:
                    title = "Sans titre"
                
                # Extraire le snippet complet
                snippet = result_element.inner_text().strip()
                
                print(f"   📝 Résultat {i+1}: {title[:50]}...")
                print(f"      🔗 URL: {link}")
                print(f"      ✅ Domaine autorisé: {is_domain_allowed(link) if link else False}")
                
                # ✅ Vérifier si l'URL est autorisée
                snippets.append(snippet)
                links.append(link)
                titles.append(title)
                    
            except Exception as e:
                print(f"⚠ Erreur lors de l'extraction du résultat {i}: {e}")
                continue

        print(f"✅ {len(snippets)} résultats valides extraits")
        
        # Si aucun résultat trouvé, essayer une recherche plus large
        if len(snippets) == 0:
            print("🔄 Tentative de recherche plus large...")
            
            # Recherche sans restriction de domaine pour debug
            broader_query = query + "faculté des sciences oujda"
            broader_url = f"https://www.google.com/search?q={broader_query}&num=5&hl={lang}"
            
            try:
                page.goto(broader_url, wait_until="networkidle", timeout=80000)
                time.sleep(2)
                
                # Réessayer avec les sélecteurs
                for selector in possible_selectors:
                    test_results = page.locator(selector)
                    if test_results.count() > 0:
                        print(f"🔍 Recherche large: {test_results.count()} résultats avec {selector}")
                        
                        # Prendre quelques exemples pour voir la structure
                        for i in range(min(3, test_results.count())):
                            try:
                                example_text = test_results.nth(i).inner_text()[:100]
                                print(f"   Exemple {i+1}: {example_text}...")
                            except:
                                pass
                        break
                        
            except Exception as e:
                print(f"⚠ Recherche large échouée: {e}")
        
        browser.close()
        
        if len(snippets) == 0:
            print("❌ Aucun résultat trouvé même avec les méthodes alternatives")
            return []
        
        # 🧠 Filtrage par pertinence sémantique
        top_snippets = filter_snippets_by_semantic_relevance(snippets, query, top_k=5)

        # Formatter les résultats
        resultats_formates = []
        for snippet_data in top_snippets:
            snippet = snippet_data['snippet']
            score = snippet_data['final_score']
            detailed_scores = snippet_data['detailed_scores']
            
            # Trouver l'index correspondant
            try:
                idx = snippets.index(snippet)
                title = titles[idx]
                link = links[idx]
                
                resultat = {
                    'titre': title,
                    'snippet': snippet,
                    'url': link,
                    'score_final': round(score, 3),
                    'scores_detailles': {k: round(v, 3) for k, v in detailed_scores.items()}
                }
                resultats_formates.append(resultat)
                
            except ValueError:
                continue

        return resultats_formates

def afficher_resultats(resultats):
    """
    Affiche les résultats de manière formatée avec les scores.
    """
    if not resultats:
        print("❌ Aucun résultat pertinent trouvé")
        return
    
    print(f"\n🏆 {len(resultats)} résultats les plus pertinents:\n")
    print("=" * 80)
    
    for i, resultat in enumerate(resultats, 1):
        print(f"\n📋 RÉSULTAT #{i} - Score: {resultat['score_final']}")
        print(f"📰 Titre: {resultat['titre']}")
        print(f"🔗 URL: {resultat['url']}")
        print(f"\n📝 Contenu:")
        print(f"{resultat['snippet'][:300]}{'...' if len(resultat['snippet']) > 300 else ''}")
        
        print(f"\n📊 Scores détaillés:")
        for metric, score in resultat['scores_detailles'].items():
            print(f"   • {metric}: {score}")
        
        print("-" * 80)

async def test(qst: str, lang : str):
    # Effectuer la recherche
    search_results = google_search_and_extract(qst, lang, max_results=15)
    
    # Formater les résultats pour le LLM
    formatted_results = []
    for i, result in enumerate(search_results, 1):
        formatted_results.append({
            'question': qst,
            'answer': f"{result['titre']}\n{result['snippet']}",
            'score': result['score_final'],
            'meta': {
                'url': result['url'],
                'detailed_scores': result['scores_detailles']
            }
        })
    
    # Passer au LLMService pour structurer la réponse

    llm_response = llm_service.generate_structured_response(
        question=qst,
        search_results=formatted_results,
        lang=lang
    )
    
    # Afficher la réponse finale structurée
    return {
        "detected_lang": lang,
        "raw_results": formatted_results,
        "structured_response": llm_response['response'],
        "confidence": llm_response['confidence'],
        "sources_used": llm_response['sources_used'],
        "processing_time": llm_response['processing_time'],
        "llm_used": True,
        "search_source": "Internet"
    }