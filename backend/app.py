
import asyncio
import os
if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

from LLMService import llm_service
from polite import is_not_defined, detect_custom_language
from typing import List, Optional, Dict, Tuple, Union
from SerpService import internet
from classifiers.classifier import MultilingualIntentClassifier
from indexer import index_faq_data



os.environ['PYTHONASYNCIODEBUG'] = '1'

LANG_MAP = {
    'fr': 'fr',
    'en': 'en',
    'ar': 'ar',    
    'amz': 'amz'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True,
        timeout=30
    )
    # Test la connexion immédiatement
    if not client.ping():
        raise HTTPException(status_code=500, detail="Impossible de se connecter à OpenSearch")
except Exception as e:
    logger.error(f"Erreur de connexion OpenSearch: {str(e)}")
    raise

try:
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
except Exception as e:
    logger.error(f"Erreur de chargement du modèle: {str(e)}")
    raise

class Query(BaseModel):
    question: str
    lang: str 
    k: int = 3
    score_threshold: float = 0.01
    use_llm: bool = True  
    context: Optional[dict] = None



logger = logging.getLogger(__name__)

@app.post("/search")
def search(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
    
    # Détection langue et vérification pertinence
    query.lang = detect_custom_language(query.question)
    
    logger.info('Must : Checking question relation with FSO')
    if not llm_service.is_faculty_related(query.question, query.lang):
        return {
            "detected_lang": query.lang,
            "structured_response": is_not_defined(query.lang),
            "llm_used": False,
            "search_source": "none"
        }
    
    if llm_service.classify_question_type(query.question, query.lang) == "dynamic":
        logger.info('Must : Searching internet')
        return internet(query.question, query.lang)
    
    try:

        classifier = MultilingualIntentClassifier('questions_intents.csv')

        # 2. Load your trained model
        classifier.load_model('classifiers/advanced_multilingual_intent_classifier.pkl')

        # Get predicted intent with probabilities
        intent, probabilities = classifier.predict_intent(query.question, return_probabilities=True)

        # Initialize the list to collect all documents
        all_documents: List[Dict[str, Union[str, float]]] = []

        # Process top 3 predicted intents and collect all documents
        for pred_intent, prob in list(probabilities.items())[:3]:
            docs = index_faq_data("dataset_dict.json", pred_intent, query.lang, prob)
            
            # Handle different return types from index_faq_data
            if isinstance(docs, list) and prob > 0.01:
                all_documents.extend(docs)  # If docs is a list, extend
            elif isinstance(docs, dict) and prob > 0.01:
                all_documents.append(docs)  # If docs is a single dict, append
            else:
                logger.warning(f"Unexpected return type from index_faq_data: {type(docs)}")

        # Log the main predicted intent
        logger.info(f"Main predicted intent: {intent}")
        logger.info(f"Top 3 intents with probabilities: {list(probabilities.items())[:3]}")
        logger.info(f"Total documents collected: {len(all_documents)}")
        logger.info(f"Type of all_documents: {type(all_documents)}")

        # Check if we found any relevant documents
        if not all_documents:
            logger.info('No documents found for predicted intents, using internet search')
            # return internet(query.question, query.lang)
        else:
            # LLM integration - pass the flattened list of documents
            if query.use_llm:
                llm_response = llm_service.generate_structured_response(
                    question=query.question,
                    search_results=all_documents,  # Pass the flattened list directly
                    lang=query.lang
                )
                check = llm_service.validate_answer_relevance(query.question, llm_response["response"])
                logger.info(llm_response["response"])
                if not check:
                    return internet(query.question, query.lang)

                # If validation passes, enhance with context if provided
                if query.context:
                    llm_response['response'] = llm_service.enhance_response_with_context(
                        llm_response['response'],
                        query.context,
                        query.lang
                    )
                # Return the FSO answer (whether enhanced or not)
                return {
                    "detected_lang": query.lang,
                    "structured_response": llm_response['response'],
                    "confidence": llm_response['confidence'],
                    "llm_used": True,
                    "search_source": "database",
                    "intents_processed": len(probabilities),
                    "documents_found": len(all_documents)
                }

    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")




@app.get("/")
def root():
    """Endpoint racine avec information sur l'API"""
    return {
        "message": "API Chatbot FAQ avec LLM et SERP intelligent",
        "version": "3.0",
        "endpoints": {
            "/search": "Recherche FAQ avec option LLM et fallback SERP intelligent",
            "/chat": "Chat avec LLM activé"
        },
        "features": [
            "Recherche sémantique dans la base de données",
            "Fallback intelligent vers recherche internet",
            "Filtrage et scoring des résultats SERP",
            "Priorisation du contenu récent",
            "Extraction de snippets pertinents",
            "Intégration LLM pour structurer les réponses"
        ]
    }


