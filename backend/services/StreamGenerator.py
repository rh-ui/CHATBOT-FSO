
from asyncio.log import logger
from http.client import HTTPException
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
# from pathlib import Path
from config import core



CLASSIFIER_MODEL = core.CLASSIFIER_MODEL
DATASET = core.DATASET



class StreamGenerator :
    def __init__(self, query: Query):
        self.query = query
        
    
    def stream_search(self):
        """
            etape 1 : detect lang 

            etape 2 : verify if question is related to fso

            etape 3 : classify intent
            
            etape 4 : simplification de la question (extract tous les de sous questions si la question est composée ) --> retourne un pair {question i , responses}
            
            etape 5 : boucler sur la liste des question/reponses
            
            |
            |--> Si type === "static" --> Etape 6   : prediction d'intent à l'aide du classifier ,  
            |                             Etape 6.1 : si probabilité > 0.05 alors rechercher les documents lié à l'intent dans dataset, 
            |                                       sinon lancer une recherche sur internet
            |
            |--> Si type === "dynamic" --> Etape 6 : lancer une recherche sur internet 
            |
            
            etape 7 : verifier si les reponses sont relevance aux questions et structure une réponse pour l'user 
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

            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyse de la question...'})}\n\n"
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
                    intent, probabilities = classifier.predict_intent(res['question'].lower(), return_probabilities=True)
                    
                    logger.info(f"Question {i+1} - Main predicted intent: {intent}")
                    logger.info(f"Question {i+1} - Top 3 intents: {list(probabilities.items())[:3]}")
                    
                    
                    for pred_intent, prob in list(probabilities.items())[:3]:
                        if prob > 0.05:
                            docs = index_faq_data(DATASET, pred_intent, self.query.lang, prob)
                            if isinstance(docs, list):
                                question_documents.extend(docs)
                            elif isinstance(docs, dict):
                                question_documents.append(docs)
                    
                    logger.info(f"Question {i+1} - Database documents found: {len(question_documents)}")
                    
                    if not question_documents:
                        logger.info(f"Question {i+1} - No database documents found, searching internet")
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Réponse non pertinente détectée, recherche sur le web...'})}\n\n"
                        
                        internet_results = get_internet_results_for_question(question_text, self.query.lang)
                        question_documents.extend(internet_results)
                        logger.info(f"Question {i+1} - Internet documents found: {len(internet_results)}")
                    
                elif question_type == 'dynamic':
                    logger.info(f"Question {i+1} - Dynamic question, using internet search")
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Réponse non pertinente détectée, recherche sur le web...'})}\n\n"
                
                    internet_results = get_internet_results_for_question(question_text, self.query.lang)
                    question_documents.extend(internet_results)
                    logger.info(f"Question {i+1} - Internet documents found: {len(internet_results)}")
                
                
                all_documents.extend(question_documents)
                
                
                question_answer_pairs.append({
                    "question": question_text,
                    "original_question": self.query.question,
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

            
            if not all_documents:
                logger.info('No documents found from any source, returning no results message')
                final_result ={
                    "detected_lang": self.query.lang,
                    "structured_response": get_no_results_message(self.query.lang),
                    "llm_used": False,
                    "search_source": "none"
                }
                yield f"data: {json.dumps({'type': 'final', 'data': final_result})}\n\n"
            
            
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
                
                yield f"data: {json.dumps({'type': 'status', 'message': 'Je suis en train de structurer votre réponse...'})}\n\n"

                
                llm_response = llm_service.generate_comprehensive_response_optimized(
                    original_question=self.query.question,
                    question_answer_pairs=relevant_pairs,
                    all_documents=[doc for pair in relevant_pairs for doc in pair['documents']],
                    lang=self.query.lang,
                    validate_and_fallback=True 
                )
                
                
                if llm_response.get('used_fallback', False):
                    logger.info("LLM detected irrelevant content, fallback to internet was used")
                    
                
                
                if self.query.context:
                    llm_response['response'] = llm_service.enhance_response_with_context(
                        llm_response['response'],
                        self.query.context,
                        self.query.lang
                    )
                
                
                sources_used = set([pair['source'] for pair in relevant_pairs])
                search_source = "mixed" if len(sources_used) > 1 else list(sources_used)[0] if sources_used else "none"
                
                final_result = {
                    "detected_lang": self.query.lang,
                    "structured_response": llm_response['response'],
                    "confidence": llm_response['confidence'],
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
            
            
            
            
            
            
              
    
  