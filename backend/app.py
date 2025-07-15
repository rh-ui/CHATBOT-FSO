from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, exceptions
import logging
from langdetect import detect
from polite import is_not_defined, Ibtissam_checks
from LLMService import llm_service  # Import du service LLM
from typing import Optional
import numpy as np

LANG_MAP = {
    'fr': 'fr',
    'en': 'en',
    'ar': 'ar',    
    'amz': 'amz',
}

def detect_custom_language(text):
    try:
        lang = detect(text)
        return LANG_MAP.get(lang, 'unknown')
    except Exception:
        return 'unknown'


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
    lang: str = "fr"
    k: int = 3
    score_threshold: float = 0.8
    use_llm: bool = True  
    context: Optional[dict] = None

@app.post("/search")
def search(query: Query):
    lang = detect_custom_language(query.question) or query.lang or "fr"

    try:
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
        
        # Generate embedding
        embedding = model.encode(query.question).tolist()
        
        # Fixed KNN query syntax for OpenSearch
        query_body = {
            "size": query.k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": query.k * 2
                                }
                            }
                        }
                    ],
                    "should": [
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
                    "filter": [{"term": {"lang": lang}}] if lang else []
                }
            }
        }

        logger.info(f"Executing search with query: {query_body}")
        response = client.search(index="faq", body=query_body)
        
        # Log the response for debugging
        logger.info(f"OpenSearch response: {response}")
        
        # Apply score threshold filtering
        hits = [hit for hit in response["hits"]["hits"] if hit["_score"] >= query.score_threshold]
        
        if hits:
            results = [
                {
                    "question": hit["_source"]["question"],
                    "answer": hit["_source"]["answer"],
                    "score": hit["_score"],
                    "meta": hit["_source"].get("meta"),
                    "search_type": "knn" if "_knn_score" in hit else "match"  # Debug info
                }
                for hit in hits
            ]
                        
            if query.use_llm:
                llm_response = llm_service.generate_structured_response(
                    question=query.question,
                    search_results=results,
                    lang=lang
                )
                
                if query.context and llm_response.get('response'):
                    enhanced_response = llm_service.enhance_response_with_context(
                        llm_response['response'],
                        query.context,
                        lang
                    )
                    llm_response['response'] = enhanced_response
                
                return {
                    "detected_lang": lang,
                    "raw_results": results,
                    "structured_response": llm_response['response'],
                    "confidence": llm_response['confidence'],
                    "sources_used": llm_response['sources_used'],
                    "processing_time": llm_response['processing_time'],
                    "llm_used": True
                }
            else:
                return {
                    "detected_lang": lang,
                    "results": results,
                    "llm_used": False
                }
        else:
            if query.use_llm:
                llm_response = llm_service.generate_structured_response(
                    question=query.question,
                    search_results=[],
                    lang=lang
                )
                
                return {
                    "detected_lang": lang,
                    "structured_response": llm_response['response'],
                    "confidence": 0.0,
                    "sources_used": 0,
                    "processing_time": llm_response['processing_time'],
                    "llm_used": True
                }
            else:
                message = is_not_defined(lang)
                results = [
                    {
                        "question": None,
                        "answer": message,
                        "score": 1.0,
                        "meta": None,
                    }
                ]
                
                return {
                    "detected_lang": lang,
                    "results": results,
                    "llm_used": False
                }

    except exceptions.NotFoundError:
        raise HTTPException(status_code=404, detail="Index 'faq' non trouvé")
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.post("/chat")
def chat_with_llm(query: Query):
    """Endpoint dédié pour le chat avec LLM (toujours activé)"""
    condition = Ibtissam_checks(query.question)
    if (condition):
        query.use_llm = True
        return search(query)
    else:
        query.use_llm = False
        return is_not_defined(query.lang)


@app.get("/health")
def health_check():
    """Vérification de l'état de santé avec test du LLM"""
    health_status = {
        "opensearch": client.ping(),
        "model_loaded": True,
        "llm_service": False
    }
    

    try:
        test_response = llm_service.generate_structured_response(
            question="Test de santé",
            search_results=[],
            lang="fr"
        )
        health_status["llm_service"] = True
    except Exception as e:
        logger.error(f"Erreur test LLM: {str(e)}")
        health_status["llm_service"] = False
    
    return health_status

@app.get("/")
def root():
    """Endpoint racine avec information sur l'API"""
    return {
        "message": "API Chatbot FAQ avec LLM",
        "version": "2.0",
        "endpoints": {
            "/search": "Recherche FAQ avec option LLM",
            "/chat": "Chat avec LLM activé",
            "/health": "Vérification de santé"
        }
    }