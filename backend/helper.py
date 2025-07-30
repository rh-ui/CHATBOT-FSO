import re
from typing import List, Set
from opensearchpy import OpenSearch, exceptions
import logging


logger = logging.getLogger(__name__)

def extract_key_entities(question: str) -> List[str]:
    question_lower = question.lower()
    
    stop_words = {
        # French stop words
        'de', 'le', 'la', 'les', 'du', 'des', 'et', 'ou', 'pour', 'dans', 'sur', 'avec', 'par', 'ce', 'ces', 'cette', 'cet',
        'un', 'une', 'aux', 'est', 'sont', 'était', 'étaient', 'sera', 'seront', 'avoir', 'être', 'fait', 'faire',
        'dit', 'dire', 'tout', 'tous', 'toute', 'toutes', 'très', 'plus', 'moins', 'bien', 'mal', 'bon', 'bonne',
        'grand', 'grande', 'petit', 'petite', 'nouveau', 'nouvelle', 'vieux', 'vieille', 'jeune', 'gros', 'grosse',
        'que', 'qui', 'quoi', 'où', 'quand', 'comment', 'pourquoi', 'combien', 'quel', 'quelle', 'quels', 'quelles',
        'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous', 'me', 'te', 'se', 'lui', 'leur', 'leurs',
        'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos',
        'si', 'mais', 'donc', 'car', 'ni', 'or', 'comme', 'depuis', 'pendant', 'avant', 'après', 'sous', 'devant',
        'derrière', 'entre', 'parmi', 'selon', 'sans', 'sauf', 'vers', 'chez', 'contre', 'malgré', 'durant',
        # English stop words  
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'now', 'then', 'here', 'there', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'once', 'because', 'if', 'while', 'during', 'before', 'after', 'above', 'below', 'between', 'through'
    }
    question_words = {
        'modules', 'module', 'matiere', 'matières', 'cours', 'formation', 'programme', 'enseignement',
        'filiere', 'filières', 'specialite', 'spécialité', 'niveau', 'année', 'semestre', 'trimestre',
        'examen', 'examens', 'note', 'notes', 'coefficient', 'credit', 'credits', 'ects',
        'professeur', 'enseignant', 'etudiant', 'étudiants', 'inscription', 'candidature',
        'conditions', 'prerequis', 'prérequis', 'objectifs', 'competences', 'compétences'
    }
    
    all_stop_words = stop_words.union(question_words)
    
    entities = []
    
    patterns = [
        r'(?:modules?\s+(?:de|du|des)\s+)([a-z0-9]+)',  # "modules de SMI"
        r'(?:filiere?s?\s+)([a-z0-9]+)',                # "filiere SMI"  
        r'(?:formation\s+)([a-z0-9]+)',                 # "formation SMI"
        r'(?:licence\s+)([a-z0-9\s]+)',                 # "licence informatique"
        r'(?:master\s+)([a-z0-9\s]+)',                  # "master data science"
        r'(?:doctorat\s+)([a-z0-9\s]+)',               # "doctorat physique"
        r'\b([a-z]{2,4}[0-9]*)\b',                      # Short codes like SMI, SMA, M1, etc.
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, question_lower)
        for match in matches:
            if isinstance(match, tuple):
                entities.extend([m.strip() for m in match if m.strip()])
            else:
                entities.append(match.strip())
    
    # Strategy 2: Extract words that are likely entity names (not in stop words)
    words = re.findall(r'\b[a-z][a-z0-9]*\b', question_lower)
    for word in words:
        if (word not in all_stop_words and 
            len(word) >= 2 and 
            not word.isdigit()):
            # Additional filters for likely academic entities
            if (len(word) <= 4 or  # Short codes like SMI, SMA
                any(char.isdigit() for char in word) or  # Contains numbers like M1, L3
                word in ['informatique', 'mathematiques', 'physique', 'chimie', 'biologie', 
                        'economie', 'gestion', 'droit', 'medecine', 'pharmacie']):  # Known subjects
                entities.append(word)
    
    # Remove duplicates and filter
    entities = list(set(entities))
    
    # Final filtering: remove very common words that might have slipped through
    final_entities = []
    for entity in entities:
        if (entity not in all_stop_words and 
            len(entity) >= 2 and
            entity not in ['www', 'com', 'org', 'net', 'edu']):  # Remove web-related terms
            final_entities.append(entity)
    
    return final_entities

def validate_entities_in_db(entities: List[str], client: OpenSearch, lang: str) -> float:
    if not entities:
        return 1.0  # No specific entities to validate
    
    found_entities = 0
    
    for entity in entities:
        # Quick check if entity exists in any document
        validation_query = {
            "size": 1,
            "_source": False,  # We only need to know if it exists
            "query": {
                "bool": {
                    "should": [
                        {"wildcard": {"question": f"*{entity}*"}},
                        {"wildcard": {"answer": f"*{entity}*"}},
                        {"term": {"meta.filier": entity}}
                    ],
                    "filter": [{"term": {"lang": lang}}] if lang else []
                }
            }
        }
        
        try:
            response = client.search(index="faq", body=validation_query)
            if response["hits"]["total"]["value"] > 0:
                found_entities += 1
                logger.info(f"Entity '{entity}' found in database")
            else:
                logger.info(f"Entity '{entity}' NOT found in database")
        except Exception as e:
            logger.warning(f"Error validating entity '{entity}': {e}")
            continue
    
    validation_score = found_entities / len(entities)
    logger.info(f"Entity validation score: {validation_score} ({found_entities}/{len(entities)})")
    return validation_score