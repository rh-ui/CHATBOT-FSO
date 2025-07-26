
import asyncio
import os
if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


from concurrent.futures import ThreadPoolExecutor
import sys
import re
import numpy as np
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, exceptions

from LLMService import llm_service
from indexer import add_single_entry
from polite import is_not_defined, Ibtissam_checks, detect_custom_language
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from SerpService import test




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
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    logger.error(f"Erreur de chargement du modèle: {str(e)}")
    raise

class Query(BaseModel):
    question: str
    lang: str 
    k: int = 3
    score_threshold: float = 0.7
    use_llm: bool = True  
    context: Optional[dict] = None



logger = logging.getLogger(__name__)

@app.post("/search")
def search(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
    query.lang = detect_custom_language(query.question)
    if not Ibtissam_checks(query.question, query.lang):
        return is_not_defined(query.lang)
    try:        
        embedding = model.encode(query.question).tolist()        
        query_body = {
            "size": query.k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": 300,
                                    "boost": 0.7
                                }
                            }
                        },
                        {
                            "match": {
                                "question": {
                                    "query": query.question,
                                    "fuzziness": "AUTO",
                                    "boost": 0.3
                                }
                            }
                        }
                    ],
                    "filter": [{"term": {"lang": query.lang}}] if query.lang else []
                }
            }
        }
        response = client.search(index="faq", body=query_body)
        hits = [hit for hit in response["hits"]["hits"] if hit["_score"] >= query.score_threshold]
        
        if hits:
            results = [
                {
                    "question": hit["_source"]["question"],
                    "answer": hit["_source"]["answer"],
                    "score": hit["_score"],
                    "meta": hit["_source"].get("meta"),
                    "search_type": "knn" if "_knn_score" in hit else "match"
                }
                for hit in hits
            ]
            
            if query.use_llm:
                llm_response = llm_service.generate_structured_response(
                    question=query.question,
                    search_results=results,
                    lang=query.lang
                ) 
                
                if llm_response['confidence'] < 19:
                    return test(query.question, query.lang)
                
                if query.context and llm_response.get('response'):
                    enhanced_response = llm_service.enhance_response_with_context(
                        llm_response['response'],
                        query.context,
                        query.lang
                    )
                return {
                    "detected_lang": query.lang,
                    "raw_results": results,
                    "structured_response": llm_response['response'],
                    "confidence": llm_response['confidence'],
                    "sources_used": llm_response['sources_used'],
                    "processing_time": llm_response['processing_time'],
                    "llm_used": True,
                    "search_source": "database"
                }        
        else: #internet
            if query.use_llm:
                test(query.question, query.lang)
                

    except exceptions.NotFoundError:
        raise HTTPException(status_code=404, detail="Index 'faq' non trouvé")
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {str(e)}")
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


