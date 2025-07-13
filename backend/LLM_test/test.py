from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import spacy
import json
import re
from urllib.parse import urlparse
from typing import List, Dict, Optional
import concurrent.futures
from datetime import datetime

# Configuration
TARGET_DOMAINS = ["ump.ma", "cg.gov.ma", "maroc.ma", ".com"]
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
TIMEOUT = 10
MAX_RESULTS = 5
MAX_WORKERS = 3
KEYWORDS = ["ouvrture", "horaire", "faculté", "université", "FSO", "Oujda"]
QUERY = "a quel heure s'ouvre fso"

# Load French language model
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("French language model not found. Installing...")
    import os
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

def is_valid_url(url: str, target_domains: List[str]) -> bool:
    """Check if URL belongs to one of the target domains."""
    try:
        domain = urlparse(url).netloc
        return any(target in domain for target in target_domains)
    except:
        return False

def search_duckduckgo(query: str, max_results: int = MAX_RESULTS) -> List[str]:
    """Search DuckDuckGo and return filtered URLs."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
            return [r['href'] for r in results if is_valid_url(r.get('href', ''), TARGET_DOMAINS)]
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

def fetch_url_content(url: str) -> Optional[str]:
    """Fetch and extract text content from a URL."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
            
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"❌ Error fetching {url}: {str(e)}")
        return None

def find_relevant_sentences(text: str, keywords: List[str]) -> List[str]:
    """Extract sentences containing any of the keywords."""
    if not text:
        return []
    
    # Clean text and split into sentences
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    return [
        sentence.strip() 
        for sentence in sentences 
        if any(re.search(rf'\b{re.escape(keyword)}\b', sentence, re.I) 
               for keyword in keywords)
    ]

def extract_person_info(sentences: List[str]) -> List[Dict]:
    """Extract person names and their contexts from sentences."""
    persons = []
    seen_names = set()
    
    for sentence in sentences:
        doc = nlp(sentence)
        for ent in doc.ents:
            if ent.label_ == "PER":
                name = ent.text.strip()
                if name not in seen_names:
                    seen_names.add(name)
                    persons.append({
                        "name": name,
                        "context": sentence,
                        "position": "doyen",
                        "faculty": "Faculté des Sciences Oujda"
                    })
    return persons

def process_url(url: str) -> Optional[Dict]:
    """Process a single URL and return extracted information."""
    print(f"\n🔗 Processing: {url}")
    content = fetch_url_content(url)
    if not content:
        return None
        
    sentences = find_relevant_sentences(content, KEYWORDS)
    persons = extract_person_info(sentences)
    
    if not persons:
        return None
        
    print(f"👤 Found {len(persons)} names in {url}")
    for person in persons:
        print(f"   - {person['name']} | Context: {person['context'][:80]}...")
    
    return {
        "url": url,
        "persons": persons,
        "source": urlparse(url).netloc,
        "date_extracted": datetime.now().isoformat()
    }

def main():
    print(f"🔍 Searching for: {QUERY}")
    urls = search_duckduckgo(QUERY)
    
    if not urls:
        print("❌ No relevant URLs found.")
        return
    
    print(f"🌐 Found {len(urls)} relevant URLs to process")
    
    # Process URLs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_url, urls))
    
    # Filter out None results
    final_results = [r for r in results if r is not None]
    
    # Save results
    with open("resultats_doyen.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(final_results)} results to 'resultats_doyen.json'")

if __name__ == "__main__":
    main()