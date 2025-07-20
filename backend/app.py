from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, exceptions
import logging
from langdetect import detect

from LLMService import llm_service
from whisper_service import whisper_service
from typing import Optional
from indexer import add_single_entry
from polite import is_not_defined, Ibtissam_checks
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

class AudioQuery(BaseModel):
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
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": 340,
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
                    "filter": [{"term": {"lang": lang}}] if lang else []
                }
            }
        }

        response = client.search(index="faq", body=query_body)
        logger.info(f"OpenSearch response: {response}")
        
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
                print("use llm if!!!!!!!!!!!!!!!\n")
                faculty_check = llm_service.is_faculty_related(query.question, lang)
                
                if faculty_check.get("is_faculty_related", False):
                    response_data = llm_service.generate_faculty_response(query.question, lang)
                    entry = llm_service.format_for_database(query.question, response_data['response'], lang)
                    insertion_result = add_single_entry(entry)
                    
                    return {
                        "detected_lang": lang,
                        "structured_response": response_data['response'],
                        "confidence": response_data['confidence'],
                        "llm_used": True,
                        "source": response_data['source'],
                        "auto_inserted": insertion_result.get("success", False)
                    }
                else:
                    fallback = llm_service.no_results_messages.get(lang, llm_service.no_results_messages['fr'])
                    return {
                        "detected_lang": lang,
                        "structured_response": fallback,
                        "confidence": 0.0,
                        "llm_used": True,
                        "source": "not_related",
                        "auto_inserted": False
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


# ENDPOINTS POUR L'AUDIO
@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """Transcrit un fichier audio en texte"""
    try:
        # Vérifier le format du fichier
        if not any(audio_file.filename.lower().endswith(ext) for ext in whisper_service.get_supported_formats()):
            raise HTTPException(
                status_code=400, 
                detail=f"Format non supporté. Formats acceptés: {whisper_service.get_supported_formats()}"
            )
        
        # Lire le contenu du fichier
        audio_content = await audio_file.read()
        
        # Transcrire l'audio
        result = whisper_service.transcribe_audio_data(audio_content, audio_file.filename)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Erreur de transcription: {result.get('error', 'Erreur inconnue')}")
        
        return {
            "success": True,
            "transcription": result["text"],
            "detected_language": result.get("language", "unknown"),
            "duration": result.get("duration", 0),
            "filename": audio_file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.post("/voice-search")
async def voice_search(
    audio_file: UploadFile = File(...),
    lang: str = Form("fr"),
    k: int = Form(3),
    score_threshold: float = Form(0.8),
    use_llm: bool = Form(True),
    whisper_language: Optional[str] = Form(None)
):
    """Recherche à partir d'un fichier audio"""
    try:
        # Étape 1: Transcrire l'audio
        audio_content = await audio_file.read()
        transcription_result = whisper_service.transcribe_audio_data(audio_content, audio_file.filename)
        
        if not transcription_result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur de transcription: {transcription_result.get('error', 'Erreur inconnue')}"
            )
        
        transcribed_text = transcription_result["text"]
        detected_lang = transcription_result.get("language", lang)
        
        # Mapper la langue détectée par Whisper vers notre système
        if detected_lang in LANG_MAP:
            detected_lang = LANG_MAP[detected_lang]
        elif detected_lang not in LANG_MAP.values():
            detected_lang = lang 
        
        # Étape 2: Effectuer la recherche avec le texte transcrit
        query = Query(
            question=transcribed_text,
            lang=detected_lang,
            k=k,
            score_threshold=score_threshold,
            use_llm=use_llm
        )
        
        search_result = search(query)
        
        # Ajouter les informations de transcription au résultat
        search_result["transcription"] = {
            "original_text": transcribed_text,
            "detected_language": transcription_result.get("language", "unknown"),
            "duration": transcription_result.get("duration", 0),
            "filename": audio_file.filename
        }
        
        return search_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la recherche vocale: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.post("/voice-chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    lang: str = Form("fr"),
    whisper_language: Optional[str] = Form(None)
):
    """Chat vocal avec LLM (toujours activé)"""
    try:
        # Transcrire l'audio
        audio_content = await audio_file.read()
        transcription_result = whisper_service.transcribe_audio_data(audio_content, audio_file.filename)
        
        if not transcription_result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur de transcription: {transcription_result.get('error', 'Erreur inconnue')}"
            )
        
        transcribed_text = transcription_result["text"]
        
        condition = Ibtissam_checks(transcribed_text)
        
        if condition:
            # Utiliser le chat avec LLM
            query = Query(
                question=transcribed_text,
                lang=lang,
                use_llm=True
            )
            chat_result = search(query)
        else:
            chat_result = {
                "detected_lang": lang,
                "structured_response": is_not_defined(lang),
                "llm_used": False
            }
        
        # Ajouter les informations de transcription
        chat_result["transcription"] = {
            "original_text": transcribed_text,
            "detected_language": transcription_result.get("language", "unknown"),
            "duration": transcription_result.get("duration", 0),
            "filename": audio_file.filename
        }
        
        return chat_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chat vocal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.get("/whisper-info")
def whisper_info():
    """Informations sur le service Whisper"""
    return whisper_service.get_model_info()


@app.get("/health")
def health_check():
    """Vérification de l'état de santé avec test du LLM et Whisper"""
    health_status = {
        "opensearch": client.ping(),
        "model_loaded": True,
        "llm_service": False,
        "whisper_service": False
    }
    
    # Test LLM
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
    
    # Test Whisper
    try:
        whisper_info = whisper_service.get_model_info()
        health_status["whisper_service"] = True
        health_status["whisper_device"] = whisper_info.get("device", "unknown")
    except Exception as e:
        logger.error(f"Erreur test Whisper: {str(e)}")
        health_status["whisper_service"] = False
    
    return health_status


@app.get("/")
def root():
    """Endpoint racine avec information sur l'API"""
    return {
        "message": "API Chatbot FAQ avec LLM et Whisper",
        "version": "3.0",
        "endpoints": {
            "/search": "Recherche FAQ avec option LLM",
            "/chat": "Chat avec LLM activé",
            "/transcribe": "Transcription audio vers texte",
            "/voice-search": "Recherche à partir d'audio",
            "/voice-chat": "Chat vocal avec LLM",
            "/whisper-info": "Informations Whisper",
            "/health": "Vérification de santé"
        },
        "supported_audio_formats": whisper_service.get_supported_formats()
    }