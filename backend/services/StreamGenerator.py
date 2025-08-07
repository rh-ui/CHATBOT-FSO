
from asyncio.log import logger
from http.client import HTTPException
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio
from Models.query import Query
from .LLMService import llm_service 
from .polite import is_not_defined, detect_custom_language
from .SerpService import get_internet_results_for_question, get_no_results_message
from classifiers.classifier import MultilingualIntentClassifier
from .helper import determine_source_type, index_faq_data
import pickle
from pathlib import Path

CLASSIFIER_MODEL = Path(__file__).parent.parent / 'classifiers' / 'intent_classifier.pkl'
DATASET = Path(__file__).parent.parent / 'data' / 'dataset_dict_date.json'

class StreamGenerator:
    def __init__(self, query: Query):
        self.query = query
        
    def stream_search(self):
        """
        etape 1 : detect lang 
        etape 2 : verify if question is related to fso
        etape 3 : classify intent
        etape 4 : determine if question is <dynamic | static>
        |
        |--> if(dynamic) -> etape 5 : serp on internet
        |
        |--> if(static) -> etape 5 :  search for question in local file based on its intent
        etape 6 : restructure response based on the returned results
        """
        
        if not self.query.question.strip():
            yield f"data: {json.dumps({'type': 'error', 'message': 'La question ne peut pas etre vide'})}\n\n"
            raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

        yield f"data: {json.dumps({'type': 'status', 'message': 'Détection de la langue en cours...'})}\n\n"
        self.query.lang = detect_custom_language(self.query.question)

        yield f"data: {json.dumps({'type': 'status', 'message': 'Veuillez pateinter je suis entrain de verifier la pertinence de votre question'})}\n\n"
        logger.info('--> Checking question relation with FSO')
        
        if not llm_service.is_faculty_related(self.query.question, self.query.lang):
            yield f"data: {json.dumps({'type': 'final', 'data': {'detected_lang': self.query.lang, 'structured_response': is_not_defined(self.query.lang), 'llm_used': False, 'search_source': 'none'}})}\n\n"
            return
            
        yield f"data: {json.dumps({'type': 'status', 'message': 'Classification du type de question...'})}\n\n"
        
        try:
            classifier = MultilingualIntentClassifier()
            classifier.load_model(CLASSIFIER_MODEL)
            
            list_results = llm_service.simplify_question(self.query.question, self.query.lang)
            logger.info(f"Simplified questions: {list_results}")

            yield f"data: {json.dumps({'type': 'status', 'message': 'Recherche dans la base de connaissances...'})}\n\n"
            
            question_answer_pairs = []
            all_documents = []
            
            for i, res in enumerate(list(list_results)):
                question_text = res['question']
                question_type = res['type']
                logger.info(f"Processing question {i+1} ({question_type}): {question_text}")
                
                question_documents = []
                intent = None
                probabilities = {}
                
                if question_type == 'static':
                    intent, probabilities = classifier.predict_intent(question_text.lower(), return_probabilities=True)
                    
                    logger.info(f"Question {i+1} - Main predicted intent: {intent}")
                    logger.info(f"Question {i+1} - Top 3 intents: {list(probabilities.items())[:3]}")
                    
                    # Try database search first
                    for pred_intent, prob in list(probabilities.items())[:3]:
                        docs = index_faq_data(DATASET, pred_intent, self.query.lang, prob)
                        if isinstance(docs, list):
                            question_documents.extend(docs)
                        elif isinstance(docs, dict):
                            question_documents.append(docs)
                    
                    logger.info(f"Question {i+1} - Database documents found: {len(question_documents)}")
                    
                    # NEW: Check conditions for internet search
                    need_internet_search = False
                    
                    # Condition 1: No database documents found
                    if not question_documents:
                        need_internet_search = True
                        logger.info(f"Question {i+1} - No database documents found, will search internet")
                    
                    # Condition 2: Check relevance of database results using LLM
                    elif question_documents:
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Vérification de la pertinence des résultats...'})}\n\n"
                        
                        try:
                            # Check if database results are relevant to the question
                            is_relevant = llm_service.check_question_answer_relevance(
                                question=question_text,
                                documents=question_documents,
                                lang=self.query.lang
                            )
                            
                            if not is_relevant:
                                need_internet_search = True
                                logger.info(f"Question {i+1} - Database results not relevant, will search internet")
                            else:
                                logger.info(f"Question {i+1} - Database results are relevant, using database only")
                                need_internet_search = False
                                
                        except Exception as e:
                            logger.error(f"Question {i+1} - Relevance check failed: {str(e)}")
                            # Fallback: assume results are relevant if check fails
                            logger.info(f"Question {i+1} - Using database results due to relevance check failure")
                    
                    # Perform internet search if needed
                    if need_internet_search:
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Recherche complémentaire sur le web...'})}\n\n"
                        
                        try:
                            internet_results = get_internet_results_for_question(question_text, self.query.lang)
                            if internet_results:  # Check if results exist
                                # Clear database results if they were not relevant
                                if question_documents and not is_relevant:
                                    question_documents = []
                                question_documents.extend(internet_results)
                                logger.info(f"Question {i+1} - Internet documents found: {len(internet_results)}")
                            else:
                                logger.warning(f"Question {i+1} - No internet results returned")
                        except Exception as e:
                            logger.error(f"Question {i+1} - Internet search failed: {str(e)}")
                    
                elif question_type == 'dynamic':
                    # Condition 3: Always search internet for dynamic questions
                    logger.info(f"Question {i+1} - Dynamic question, using internet search")
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Question dynamique détectée, recherche sur le web...'})}\n\n"
                    
                    try:
                        internet_results = get_internet_results_for_question(question_text, self.query.lang)
                        if internet_results:
                            question_documents.extend(internet_results)
                            logger.info(f"Question {i+1} - Internet documents found: {len(internet_results)}")
                        else:
                            logger.warning(f"Question {i+1} - No internet results for dynamic question")
                    except Exception as e:
                        logger.error(f"Question {i+1} - Internet search failed for dynamic question: {str(e)}")
                
                # Collect all documents
                all_documents.extend(question_documents)
                
                # Create question-answer pair
                question_answer_pairs.append({
                    "question": question_text,
                    "original_question": self.query.question,
                    "intent": intent,
                    "probabilities": probabilities,
                    "documents": question_documents,
                    "question_index": i + 1,
                    "type": question_type,
                    "reason": res.get('reason', ''),
                    "source": "database" if question_type == 'static' else "internet"
                })

            logger.info(f"Total questions processed: {len(question_answer_pairs)}")
            logger.info(f"Total documents collected: {len(all_documents)}")

            # FIXED: Handle case when no documents found
            if not all_documents:
                logger.info('No documents found from any source, returning no results message')
                final_result = {
                    "detected_lang": self.query.lang,
                    "structured_response": get_no_results_message(self.query.lang),
                    "llm_used": False,
                    "search_source": "none"
                }
                yield f"data: {json.dumps({'type': 'final', 'data': final_result})}\n\n"
                return 
            
            # Process with LLM if enabled
            if self.query.use_llm:
                relevant_pairs = [pair for pair in question_answer_pairs if pair['documents']]
                
                if not relevant_pairs:
                    logger.info('No questions with documents found, returning no results')
                    final_result = {
                        "detected_lang": self.query.lang,
                        "structured_response": get_no_results_message(self.query.lang),
                        "llm_used": False,
                        "search_source": "none"
                    }
                    yield f"data: {json.dumps({'type': 'final', 'data': final_result})}\n\n"
                    return  # FIXED: Added return statement
                
                
                try:
                    llm_response = llm_service.generate_comprehensive_response_optimized(
                        original_question=self.query.question,
                        question_answer_pairs=relevant_pairs,
                        all_documents=[doc for pair in relevant_pairs for doc in pair['documents']],
                        lang=self.query.lang,
                        validate_and_fallback=True 
                    )
                except Exception as e:
                    logger.error(f"LLM response generation failed: {str(e)}")
                    # Fallback to basic response
                    llm_response = {
                        'response': get_no_results_message(self.query.lang),
                        'confidence': 0.0,
                        'used_fallback': True
                    }
                
                # Handle fallback message
                if llm_response.get('used_fallback', False):
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Utilisation du fallback pour la réponse...'})}\n\n"

                
                # Enhance with context if provided
                if self.query.context:
                    try:
                        llm_response['response'] = llm_service.enhance_response_with_context(
                            llm_response['response'],
                            self.query.context,
                            self.query.lang
                        )
                    except Exception as e:
                        logger.error(f"Context enhancement failed: {str(e)}")
                
                # Determine sources used
                sources_used = set([pair['source'] for pair in relevant_pairs if pair['source']])
                search_source = "mixed" if len(sources_used) > 1 else list(sources_used)[0] if sources_used else "none"
                
                final_result = {
                    "detected_lang": self.query.lang,
                    "structured_response": llm_response['response'],
                    "confidence": llm_response.get('confidence', 0.0),
                    "llm_used": True,
                    "search_source": search_source,
                    "original_question": self.query.question,
                    "questions_processed": len(list_results),
                    "relevant_questions_processed": len(relevant_pairs),
                    "documents_found": len([doc for pair in relevant_pairs for doc in pair['documents']]),
                    "llm_calls_made": 2 + (1 if self.query.context else 0),
                    "question_coverage": llm_response.get('question_coverage', {}),
                    "sources_breakdown": {
                        "database": len([p for p in relevant_pairs if p['source'] == 'database']),
                        "internet": len([p for p in relevant_pairs if p['source'] in ['internet', 'internet_fallback']]),
                        "mixed": len([p for p in relevant_pairs if p['source'] == 'mixed'])
                    },
                    "used_fallback": llm_response.get('used_fallback', False)
                }
                yield f"data: {json.dumps({'type': 'final', 'data': final_result})}\n\n"

        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'data': get_no_results_message(self.query.lang)})}\n\n"
            raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")



def debug_internet_search_for_question(question_text, lang):
    """Debug version of internet search to help identify issues"""
    logger.info(f"DEBUG: Starting internet search for: '{question_text}' in language: {lang}")
    
    try:
        results = get_internet_results_for_question(question_text, lang)
        logger.info(f"DEBUG: Internet search returned {len(results) if results else 0} results")
        
        if results:
            for i, result in enumerate(results[:3]):  # Log first 3 results
                logger.info(f"DEBUG: Result {i+1}: {result.get('title', 'No title')[:50]}...")
        else:
            logger.warning(f"DEBUG: No internet results found for question: {question_text}")
            
        return results
    except Exception as e:
        logger.error(f"DEBUG: Internet search failed with error: {str(e)}")
        return []

