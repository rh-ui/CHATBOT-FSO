
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
from SerpService import get_internet_results_for_question, get_no_results_message
from classifiers.classifier import MultilingualIntentClassifier
from indexer import index_faq_data
from helper import determine_source_type


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


# try:
#     client = OpenSearch(
#         hosts=[{"host": "localhost", "port": 9200}],
#         http_compress=True,
#         timeout=30
#     )
#     # Test la connexion immédiatement
#     if not client.ping():
#         raise HTTPException(status_code=500, detail="Impossible de se connecter à OpenSearch")
# except Exception as e:
#     logger.error(f"Erreur de connexion OpenSearch: {str(e)}")
#     raise

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
    
    try:
        classifier = MultilingualIntentClassifier()
        classifier.load_model('classifiers/intent_classifier.pkl')

        # Get simplified questions with enhanced metadata (LLM Call #1)
        list_results = llm_service.simplify_question(query.question, query.lang)
        logger.info(f"Simplified questions: {list_results}")

        # Enhanced data structure to track question-answer relationships
        question_answer_pairs = []
        all_documents = []
        
        # Process ALL questions (both static and dynamic) - NO LLM CALLS HERE
        for i, res in enumerate(list(list_results)):
            question_text = res['question']
            question_type = res['type']
            logger.info(f"Processing question {i+1} ({question_type}): {question_text}")
            
            question_documents = []
            intent = None
            probabilities = {}
            
            if question_type == 'static':
                # Process static questions with intent classification (NO LLM)
                intent, probabilities = classifier.predict_intent(res['question'].lower(), return_probabilities=True)
                
                logger.info(f"Question {i+1} - Main predicted intent: {intent}")
                logger.info(f"Question {i+1} - Top 3 intents: {list(probabilities.items())[:3]}")
                
                # Collect documents for this specific question
                for pred_intent, prob in list(probabilities.items())[:3]:
                    if prob > 0.05:
                        docs = index_faq_data("dataset_dict_date.json", pred_intent, query.lang, prob)
                        if isinstance(docs, list):
                            question_documents.extend(docs)
                        elif isinstance(docs, dict):
                            question_documents.append(docs)
                
                logger.info(f"Question {i+1} - Database documents found: {len(question_documents)}")
                
                # If no relevant documents found in database, use internet search
                if not question_documents:
                    logger.info(f"Question {i+1} - No database documents found, searching internet")
                    internet_results = get_internet_results_for_question(question_text, query.lang)
                    question_documents.extend(internet_results)
                    logger.info(f"Question {i+1} - Internet documents found: {len(internet_results)}")
                
            elif question_type == 'dynamic':
                # For dynamic questions, directly use internet search
                logger.info(f"Question {i+1} - Dynamic question, using internet search")
                internet_results = get_internet_results_for_question(question_text, query.lang)
                question_documents.extend(internet_results)
                logger.info(f"Question {i+1} - Internet documents found: {len(internet_results)}")
            
            # Add all documents to global collection
            all_documents.extend(question_documents)
            
            # Store question-answer pair with metadata
            question_answer_pairs.append({
                "question": question_text,
                "original_question": query.question,
                "intent": intent,
                "probabilities": probabilities,
                "documents": question_documents,
                "question_index": i + 1,
                "type": question_type,
                "reason": res.get('reason', ''),
                "source": determine_source_type(question_type, question_documents)
            })

        logger.info(f"Total questions processed: {len(question_answer_pairs)}")
        logger.info(f"Total documents collected: {len(all_documents)}")

        # Check if we found any relevant documents at all
        if not all_documents:
            logger.info('No documents found from any source, returning no results message')
            return {
                "detected_lang": query.lang,
                "structured_response": get_no_results_message(query.lang),
                "llm_used": False,
                "search_source": "none"
            }
        
        # Enhanced LLM integration with comprehensive context
        if query.use_llm:
            # Filter out questions with no documents (simple filtering, no LLM)
            relevant_pairs = [pair for pair in question_answer_pairs if pair['documents']]
            
            if not relevant_pairs:
                logger.info('No questions with documents found, returning no results')
                return {
                    "detected_lang": query.lang,
                    "structured_response": get_no_results_message(query.lang),
                    "llm_used": False,
                    "search_source": "none"
                }
            
            # SINGLE LLM CALL for comprehensive response generation (LLM Call #2)
            llm_response = llm_service.generate_comprehensive_response_optimized(
                original_question=query.question,
                question_answer_pairs=relevant_pairs,
                all_documents=[doc for pair in relevant_pairs for doc in pair['documents']],
                lang=query.lang,
                validate_and_fallback=True  # Built-in validation and fallback
            )
            
            # If validation failed and fallback was used, update search source
            if llm_response.get('used_fallback', False):
                # Additional internet search was performed, update pairs
                logger.info("LLM detected irrelevant content, fallback to internet was used")
            
            # Context enhancement if provided (LLM Call #3 - optional)
            if query.context:
                llm_response['response'] = llm_service.enhance_response_with_context(
                    llm_response['response'],
                    query.context,
                    query.lang
                )
            
            # Determine search sources used
            sources_used = set([pair['source'] for pair in relevant_pairs])
            search_source = "mixed" if len(sources_used) > 1 else list(sources_used)[0] if sources_used else "none"
            
            return {
                "detected_lang": query.lang,
                "structured_response": llm_response['response'],
                "confidence": llm_response['confidence'],
                "llm_used": True,
                "search_source": search_source,
                "original_question": query.question,
                "questions_processed": len(list_results),
                "relevant_questions_processed": len(relevant_pairs),
                "documents_found": len([doc for pair in relevant_pairs for doc in pair['documents']]),
                "llm_calls_made": 2 + (1 if query.context else 0),  # Track LLM calls
                "question_coverage": llm_response.get('question_coverage', {}),
                "sources_breakdown": {
                    "database": len([p for p in relevant_pairs if p['source'] == 'database']),
                    "internet": len([p for p in relevant_pairs if p['source'] in ['internet', 'internet_fallback']]),
                    "mixed": len([p for p in relevant_pairs if p['source'] == 'mixed'])
                },
                "used_fallback": llm_response.get('used_fallback', False)
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


