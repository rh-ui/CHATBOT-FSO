import json
import re
import logging
from typing import Dict, List
logging.basicConfig(level=logging.INFO)



logger = logging.getLogger(__name__)

def index_faq_data(dict_file, intent_val, lang, confidence): #using this
    # Load the dict structure
    with open(dict_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Get the exact entry in O(1)
    entry = dataset.get(intent_val)
    if not entry:
        print(f"No entry found for intent: {intent_val}")
        return []

    reponses = entry.get("reponse", {})
    metas = entry.get("meta", {})  # Some entries may not have this
    date = entry.get("data", str)

    if lang not in reponses:
        print(f"No responses found for lang: {lang}")
        return []

    docs = []
    for answer in reponses[lang]:
        doc = {
            "answer": answer,
            "lang": lang,
            "intent": intent_val,
            "confidence": confidence,
            "date": date
        }
        if metas and lang in metas and metas[lang]:
            doc["meta"] = metas[lang][0]

        docs.append(doc)

    return docs

def filter_fso_content(serp_data: list[dict]) -> list[dict]: #using this
    """Filtre le contenu SERP pour garder seulement les données FSO"""
    
    # Debug: Print input data
    print(f"DEBUG: Received {len(serp_data) if serp_data else 0} search results")
    if serp_data:
        print(f"DEBUG: First result type: {type(serp_data[0])}")
        if isinstance(serp_data[0], dict):
            print(f"DEBUG: First result keys: {list(serp_data[0].keys())}")
            print(f"DEBUG: First result snippet type: {type(serp_data[0].get('snippet', 'N/A'))}")
    
    # Handle empty or invalid input
    if not serp_data or not isinstance(serp_data, list):
        print("DEBUG: No valid input data")
        return []
    
    filtered_content = []
    
    for i, serp_item in enumerate(serp_data, 1):
        # Skip if not a dictionary or missing required fields
        if not isinstance(serp_item, dict):
            continue
            
        # Get snippet text, handle missing snippet or different types
        snippet = serp_item.get("snippet", "")
        
        # Convert snippet to string if it's not already
        if isinstance(snippet, list):
            snippet = " ".join(str(item) for item in snippet)
        elif not isinstance(snippet, str):
            snippet = str(snippet) if snippet is not None else ""
            
        if not snippet.strip():
            continue
            
        # Convert to lowercase for case-insensitive matching
        snippet_lower = snippet.lower()
        
        # Indicateurs négatifs (autres facultés)
        fso_negative = [
            "faculté des lettres", "flsh", "économie", "Faculté des Lettres et Sciences Humaines", "National School of Business and Management",
            "École Supérieure de Technologie", "encg", "droits", "lettres",
            "esto", "fsjes", "fpn", "flsh", "Sciences Humaines","École nationale de commerce et de gestion",
            "ensao", "encgo", "est", "eniad", "École Nationale des Sciences Appliquées", 
            "juridiques", "économiques", "économiques et sociales", "Faculté Pluridisciplinaire",
            "fmpo", "pluridisciplinaire","nador", "École Nationale de l'Intelligence Artificielle et du Digital",
            "droit", "Faculté des Sciences Juridiques, Économiques et Sociales",

            #en
            "National School of Business and Management"
        ]
        
        # Vérifier indicateurs négatifs (case-insensitive)
        has_negative = any(neg_term in snippet_lower for neg_term in fso_negative)
        
        # Debug: Print filtering decision
        if has_negative:
            print(f"DEBUG: Filtered out result {i+1}: contains negative terms")
        else:
            print(f"DEBUG: Keeping result {i+1}: {snippet[:200]}...")
        
        if not has_negative:
            # Update the snippet in the result to ensure it's a string
            serp_item_copy = serp_item.copy()
            serp_item_copy['snippet'] = snippet
            filtered_content.append(serp_item_copy)
        
        # Limite la taille pour éviter les timeouts
        if len(filtered_content) >= 2000:
            break
    
    # Log the filtering results
    try:
        logger.info(f"Filtered SERP content: {len(serp_data)} -> {len(filtered_content)} items")
    except NameError:
        # If logger is not available, use print
        print(f"Filtered SERP content: {len(serp_data)} -> {len(filtered_content)} items")
    
    return filtered_content

