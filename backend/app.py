from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, exceptions
import logging
from langdetect import detect
from polite import is_not_defined




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
    

# Configuration du logging
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



# Configuration avec gestion d'erreur améliorée
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
    score_threshold: float = 0.5


@app.post("/search")

def search(query: Query):
    lang = detect_custom_language(query.question) or query.lang or "fr" 
    try:
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
        
        embedding = model.encode(query.question).tolist()
        
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        {"knn": {"embedding": {"vector": embedding, "k": query.k}}},
                        {"match": {"question": {"query": query.question, "fuzziness": "AUTO"}}}
                    ],
                    "filter": [{"term": {"lang": lang}}] if lang else []
                }
            },
            "size": query.k
        }

        response = client.search(index="faq", body=query_body)
        hits = [hit for hit in response["hits"]["hits"] if hit["_score"] >= query.score_threshold]
        
        if hits:
            results = [
                {
                    "question": hit["_source"]["question"],
                    "answer": hit["_source"]["answer"],
                    "score": hit["_score"],
                    "meta": hit["_source"].get("meta")
                }
                for hit in hits
            ]
        else:
            message = is_not_defined(query.lang)

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
             "results": results
        }

    except exceptions.NotFoundError:
        raise HTTPException(status_code=404, detail="Index 'faq' non trouvé")
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/health")

def health_check():
    return {
        "opensearch": client.ping(),
        "model_loaded": True
    }