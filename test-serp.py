import os
import sys
import logging
import traceback
import requests
import json
from pathlib import Path
from polite import Ibtissam_checks,detect_custom_language

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base URL of your FastAPI application
BASE_URL = "http://localhost:8000"  # Change this if your app runs on a different URL

def serp_service(question: str):
    """Test the SERP endpoint and verify file storage with question preservation"""
    try:
        # Clear any existing test file
        test_file = Path("test.json")
        
        if not Ibtissam_checks(question):
            return
        
        lang = detect_custom_language(question)

        
        # Prepare the query data
        query_data = {
            "question": question,
            "lang": lang,
            "use_llm": True
        }
        
        
        # Make the POST request to the /test-serp endpoint
        response = requests.post(f"{BASE_URL}/test-serp", json=query_data)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            print("\n✓ API Response received successfully")
            print(f"Response keys: {list(result.keys())}")
            
            # Verify original question is preserved in the response
            if 'original_question' in result:
                if result['original_question'] == query_data.question:
                    print("✓ Original question preserved in response")
                else:
                    print(f"✗ Original question changed: '{result['original_question']}'")
            
            
            # Check if file was created and contains data
            if test_file.exists():
                print("\n✓ test.json file was created")
                
                with open(test_file, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                
                print(f"✓ File contains {len(stored_data)} entries")
                
                if stored_data:
                    latest_entry = stored_data[-1]  # Get the latest entry
                    print(f"\n--- LATEST STORED ENTRY ---")
                    print(f"ID: {latest_entry.get('id', 'N/A')}")
                    print(f"Timestamp: {latest_entry.get('timestamp', 'N/A')}")
                    print(f"Source: {latest_entry.get('source', 'N/A')}")
                    print(f"Intent: {latest_entry.get('intent', 'N/A')}")
                    
                    # Check question preservation in stored data
                    questions = latest_entry.get('question', {})
                    if questions.get('fr'):
                        stored_fr_question = questions['fr'][0] if questions['fr'] else "None"
                        print(f"Stored FR question: '{stored_fr_question}'")
                        
                        if stored_fr_question == query_data.question:
                            print("✓ Original question correctly preserved in storage")
                        else:
                            print(f"✗ Original question was changed in storage!")
                            print(f"  Original: '{query_data.question}'")
                            print(f"  Stored:   '{stored_fr_question}'")
                    
                    # Check multilingual structure
                    answers = latest_entry.get('reponse', {})
                    
                    print(f"Languages in questions: {list(questions.keys()) if questions else 'None'}")
                    print(f"Languages in answers: {list(answers.keys()) if answers else 'None'}")
                    
                    if answers.get('fr'):
                        print(f"Sample French answer: {answers['fr'][0][:150]}...")
                        
                        # Check if answer seems relevant
                        answer_lower = answers['fr'][0].lower()
                        answer_keywords = [kw for kw in query_data.question if kw in answer_lower]
                        if answer_keywords:
                            print(f"✓ Answer contains relevant keywords: {answer_keywords}")
                        else:
                            print("⚠ Answer might not be relevant to the question")
                
            else:
                print("✗ test.json file was not created")
                
            # Display confidence and other metrics
            if 'confidence' in result:
                print(f"\nConfidence: {result['confidence']}")
                
            if 'stored_to_file' in result:
                print(f"Stored to file: {result['stored_to_file']}")
                
            if 'processing_time' in result:
                print(f"Processing time: {result['processing_time']:.2f} seconds")
                
        else:
            print(f"✗ API request failed with status code: {response.status_code}")
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error testing SERP endpoint: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    serp_service('qui le fondateur de la faculté de science oujda ?')

##################################################################################################################

@app.post("/test-serp")
async def test_serp_endpoint(query: Query):
    """
    Enhanced SERP endpoint using GoogleSearchService class
    """
    
    # Input validation
    if not query.question or not query.question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty",
            "query": "",
            "response": "Veuillez fournir une question valide.",
            "confidence": 0.0,
            "debug": {
                "validation_error": True,
                "error_type": "empty_query"
            }
        }
    

    # Detect language if not provided
    detected_lang = detect_custom_language(query.question) if not query.lang else query.lang
    
    def run_sync_search():
        """
        Synchronous search function to run in thread executor
        Uses the enhanced GoogleSearchService class
        """
        try:
            
            query.use_llm = True
            from brahim_serp.serp_service import GoogleSearchService
            
            # Create service instance
            search_service = GoogleSearchService()
            
            # Perform search with retry logic
            results = search_service.search(
                query=query.question,
                lang=detected_lang
            )
            
            return results
            
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            return {"error": "service_import_failed", "details": str(e)}
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"error": "search_failed", "details": str(e)}
    
    try:
        # Run the synchronous search in a thread executor to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_sync_search)
            
            # Wait for completion with timeout
            try:
                serp_results = future.result(timeout=220)  # 60 second timeout
            except TimeoutError:
                logger.error("Search operation timed out")
                return {
                    "status": "error",
                    "message": "Search operation timed out",
                    "query": query.question,
                    "response": "La recherche a pris trop de temps. Veuillez réessayer.",
                    "confidence": 0.0,
                    "debug": {
                        "timeout_error": True,
                        "timeout_duration": 60
                    }
                }
        
        # Handle search errors
        if isinstance(serp_results, dict) and "error" in serp_results:
            error_type = serp_results.get("error", "unknown")
            error_details = serp_results.get("details", "")
            
            logger.error(f"Search service error: {error_type} - {error_details}")
            
            return {
                "status": "error",
                "message": f"Search service error: {error_type}",
                "query": query.question,
                "response": "Une erreur s'est produite lors de la recherche. Veuillez réessayer.",
                "confidence": 0.0,
                "debug": {
                    "service_error": True,
                    "error_type": error_type,
                    "error_details": error_details
                }
            }
        
        # Handle no results from search
        if not serp_results:
            logger.warning("Google search returned no results")
            return {
                "status": "no_results",
                "message": "Google search returned no results",
                "query": query.question,
                "response": "Aucun résultat trouvé pour cette recherche.",
                "confidence": 0.0,
                "debug": {
                    "search_completed": True,
                    "results_found": False
                }
            }
        
        # Check if search results contain meaningful content
        has_business_info = bool(serp_results.get("business_info", "").strip())
        has_schedule_info = bool(serp_results.get("schedule_info", "").strip())
        has_search_results = bool(serp_results.get("search_results", "").strip())
        
        has_content = has_business_info or has_schedule_info or has_search_results
        
        if not has_content:
            logger.warning("Search completed but no meaningful content found")
            return {
                "status": "no_content",
                "message": "Search completed but no relevant content found",
                "query": query.question,
                "response": "La recherche n'a pas trouvé d'informations pertinentes.",
                "confidence": 0.1,
                "debug": {
                    "search_completed": True,
                    "content_found": False,
                    "serp_structure_valid": True
                }
            }
        
        logger.info("Processing search results with LLM...")
        
        # Process results with LLM
        try:

            llm_output = llm_service.process_serp_to_response(
                question=query.question,
                serp_data=serp_results,
                lang=detected_lang
            )
            
            # Extract response and confidence
            response_text = llm_output.get('display', '') or llm_output.get('response', '')
            confidence = float(llm_output.get('confidence', 0.0))
            
            # Validate LLM output
            if response_text and len(response_text.strip()) > 0:
                logger.info("LLM processing successful")
                return {
                    "status": "success",
                    "query": query.question,
                    "detected_lang": detected_lang,
                    "response": response_text.strip(),
                    "confidence": confidence,
                    "source": "google_serp_llm",
                    "debug": {
                        "search_completed": True,
                        "llm_processing_success": True,
                        "content_types": {
                            "business_info": has_business_info,
                            "schedule_info": has_schedule_info,
                            "search_results": has_search_results
                        },
                        "content_lengths": {
                            "business_info": len(serp_results.get("business_info", "")),
                            "schedule_info": len(serp_results.get("schedule_info", "")),
                            "search_results": len(serp_results.get("search_results", ""))
                        }
                    }
                }
            else:
                logger.warning("LLM returned empty response")
                raise ValueError("LLM returned empty response")
                
        except Exception as llm_error:
            logger.error(f"LLM processing failed: {str(llm_error)}")
            
            # Fallback: Create response from raw scraped data
            fallback_response = ""
            fallback_confidence = 0.4
            
            if has_business_info:
                business_preview = serp_results["business_info"][:400]
                fallback_response = f"Informations trouvées: {business_preview}"
                if len(serp_results["business_info"]) > 400:
                    fallback_response += "..."
                    
            elif has_schedule_info:
                schedule_preview = serp_results["schedule_info"][:400]
                fallback_response = f"Horaires trouvés: {schedule_preview}"
                if len(serp_results["schedule_info"]) > 400:
                    fallback_response += "..."
                    
            elif has_search_results:
                results_preview = serp_results["search_results"][:400]
                fallback_response = f"D'après les résultats de recherche: {results_preview}"
                if len(serp_results["search_results"]) > 400:
                    fallback_response += "..."
            else:
                fallback_response = "Des informations ont été trouvées mais sont difficiles à traiter."
                fallback_confidence = 0.2
            
            return {
                "status": "partial_success",
                "message": "Search successful but LLM processing failed",
                "query": query.question,
                "detected_lang": detected_lang,
                "response": fallback_response,
                "confidence": fallback_confidence,
                "source": "google_serp_fallback",
                "debug": {
                    "search_completed": True,
                    "llm_processing_success": False,
                    "llm_error": str(llm_error),
                    "fallback_used": True,
                    "content_types": {
                        "business_info": has_business_info,
                        "schedule_info": has_schedule_info,
                        "search_results": has_search_results
                    }
                }
            }
        
    except TimeoutError:
        # This is already handled above, but keeping for extra safety
        logger.error("Operation timeout")
        return {
            "status": "error",
            "message": "Operation timed out",
            "query": query.question,
            "response": "L'opération a pris trop de temps. Veuillez réessayer.",
            "confidence": 0.0,
            "debug": {
                "timeout_error": True
            }
        }
        
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = str(e)
        logger.error(f"Unexpected error in test-serp endpoint: {error_msg}")
        
        return {
            "status": "error", 
            "message": "An unexpected error occurred",
            "query": query.question,
            "response": "Une erreur inattendue s'est produite. Veuillez réessayer.",
            "confidence": 0.0,
            "debug": {
                "unexpected_error": True,
                "error_details": error_msg,
                "error_type": type(e).__name__
            }
        }


