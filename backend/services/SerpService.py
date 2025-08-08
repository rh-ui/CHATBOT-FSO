import logging
from typing import Dict, List
from playwright.sync_api import sync_playwright
import time
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from ..services.LLMService import llm_service



logger = logging.getLogger(__name__)

WHITELIST_DOMAINS = ["fso.ump.ma", ".gov.ma"]

def is_domain_allowed(url):
    return any(domain in url for domain in WHITELIST_DOMAINS)

def extract_keywords(text):
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

    scores = {}
    
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([query, snippet])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        scores['tfidf'] = tfidf_score
    except:
        scores['tfidf'] = 0.0
    
    query_keywords = set(extract_keywords(query))
    snippet_keywords = set(extract_keywords(snippet))
    
    if query_keywords:
        keyword_match = len(query_keywords.intersection(snippet_keywords)) / len(query_keywords)
        scores['keyword_match'] = keyword_match
    else:
        scores['keyword_match'] = 0.0
    
    snippet_words = extract_keywords(snippet)
    if snippet_words:
        keyword_density = sum(1 for word in snippet_words if word in query_keywords) / len(snippet_words)
        scores['keyword_density'] = keyword_density
    else:
        scores['keyword_density'] = 0.0
    
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
    
    optimal_length = 150  # Longueur optimale d'un snippet
    length_penalty = abs(len(snippet) - optimal_length) / optimal_length
    length_score = max(0.0, 1.0 - length_penalty)
    scores['length'] = length_score
    
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
    logger.info("enter google_search_and_extract")
    with sync_playwright() as p:
        
        logger.info("enter sync_playwright")

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

        if (lang == 'fr'):
            if 'fso' in query.lower():
                query = query.replace('fso', 'la faculté des science oujda')
        elif lang == 'en':
            if 'fso' in query.lower():
                query['fso'] = query.get('fso', '').replace('faculty of science oujda', '')

        elif lang == 'ar':
            if 'fso' in query.lower():
                query['fso'] = query.get('fso', '').replace('كلية العلوم وجدة', '')

        elif lang == 'amz':
            if 'fso' in query.lower():
                query['fso'] = query.get('fso', '').replace('la faculté des science oujda', '')
        filtered_query = f"site:fso.ump.ma OR site:cg.gov.ma {query}"

        search_url = f"https://www.google.com/search?q={filtered_query}&num={max_results}&hl={lang}"

        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
            
        except Exception as e:
            logger.error(e)
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
                    if consent_button.is_visible(timeout=2000):
                        consent_button.click()
                        page.wait_for_load_state('networkidle')
                        break
                except:
                    continue
        except Exception:
            logger.error(Exception)
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
                    
                    # Wait for user to solve CAPTCHA (check if search results appear)
                    for i in range(60):  # Wait up to 60 seconds
                        try:
                            if page.locator('div#search, div#rso').is_visible(timeout=1000):
                                break
                        except:
                            pass
                        time.sleep(1)
                        if i == 59:
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
            logger.info("div#search not found, trying div#rso")
            try:
                page.wait_for_selector('div#rso', timeout=5000)
            except Exception:
                logger.info("No search results found, checking page content...")

        # Optional extra wait for page to stabilize
        time.sleep(2)

        logger.info("🔍 Recherche des sélecteurs possibles...")
        
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
            logger.info(f"  🔍 {selector}: {count} éléments trouvés")
            
            if count > 0:
                results = test_results
                working_selector = selector
                break
        
        if not results or results.count() == 0:
            try:
                page.screenshot(path="debug_google_results.png")
            except e:
                logger.info(e)
                pass
            
            # Essayer de trouver tous les liens
            all_links = page.locator("a[href*='http']")
            browser.close()
            return []

        snippets = []
        links = []
        titles = []
        
        count = min(results.count(), max_results)
        
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
                
                snippets.append(snippet)
                links.append(link)
                titles.append(title)
                    
            except Exception as e:
                logger.info(e)
                continue

        if len(snippets) == 0:
            broader_query = query
            broader_url = f"https://www.google.com/search?q={broader_query}&num=5&hl={lang}"
            
            try:
                page.goto(broader_url, wait_until="networkidle", timeout=80000)
                time.sleep(2)
                
                for selector in possible_selectors:
                    test_results = page.locator(selector)
                    if test_results.count() > 0:
                        
                        # Prendre quelques exemples pour voir la structure
                        for i in range(min(3, test_results.count())):
                            try:
                                example_text = test_results.nth(i).inner_text()[:100]
                                logger.info(f"   Exemple {i+1}: {example_text}...")
                            except:
                                pass
                        break
                        
            except Exception as e:
                logger.info(f"Recherche large échouée: {e}")
        
        browser.close()
        
        if len(snippets) == 0:
            return []
        
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
                logger.info(ValueError)
                continue

        return resultats_formates

def afficher_resultats(resultats):

    if not resultats:
        return
    
    for i, resultat in enumerate(resultats, 1):
        for metric, score in resultat['scores_detailles'].items():
            logger.info(f"   • {metric}: {score}")
        
        logger.info("-" * 80)

def get_no_results_message(lang: str) -> str:
    """Get appropriate no results message based on language"""
    messages = {
        'fr': "Désolé, je n'ai pas trouvé d'informations pertinentes pour répondre à votre question. Pouvez-vous reformuler votre question ou être plus spécifique?",
        'en': "Sorry, I couldn't find relevant information to answer your question. Could you rephrase your question or be more specific?",
        'ar': "عذراً، لم أتمكن من العثور على معلومات ذات صلة للإجابة على سؤالك. هل يمكنك إعادة صياغة سؤالك أو أن تكون أكثر تحديداً؟",
        'amz': "Suref-iyi, ur ufiɣ ara talɣut yesɛan azday i tririt n usqsi-inek. Tzemreḍ ad talseḍ asqsi-inek neɣ ad tiliḍ d-aweḥḥed?"
    }
    return messages.get(lang, messages['fr'])

def get_internet_results_for_question(question: str, lang: str) -> List[Dict]:
    """
    Get internet search results for a specific question and format them consistently
    NO LLM CALLS HERE - Just data formatting
    """
    try:
        logger.info(f"Searching internet for: {question}")
        search_results = google_search_and_extract(question, lang, max_results=15)
        logger.info(f"Serp-search_results-> {search_results}")
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            formatted_result = {
                'question': question,
                'answer': f"{result['titre']}\n{result['snippet']}",
                'lang': lang,
                'intent': 'internet_search',
                'confidence': result['score_final'],
                'date': 'recent',
                'source': 'internet',
                'meta': {
                    'url': result['url'],
                    'detailed_scores': result['scores_detailles'],
                    'title': result['titre']
                }
            }
            formatted_results.append(formatted_result)
        
        logger.info(f"Internet search returned {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in internet search: {str(e)}", exc_info=True)
        return []