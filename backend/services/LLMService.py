import requests
import logging
from typing import List, Dict, Any, Union
from pydantic import BaseModel
import json
import os
from datetime import datetime
import psutil
import GPUtil
from .SerpService import get_internet_results_for_question


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
        os.environ['OLLAMA_GPU_LAYERS'] = '999'
        os.environ['OLLAMA_NUM_PARALLEL'] = '1'
        os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'
        os.environ['OLLAMA_KEEP_ALIVE'] = '10m'
        
        
        os.environ['NVIDIA_VISIBLE_DEVICES'] = '1'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        
        self.base_url = "http://localhost:11434"
        self.model_name = "llama3:8b"
        
        
        self.gpu_optimized_options = {
            "num_ctx": 2048, 
            "num_batch": 512,  
            "num_gqa": 8,     
            "num_gpu": 999,  
            "num_thread": 4,   
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "num_predict": 1200,
            "use_mmap": True,   
            "use_mlock": True,
        }

        
        self.prompts = {
            'fr': {
                'system': """Tu es l'assistant virtuel officiel de la Faculté des Sciences d'Oujda (FSO). 
                
                Ton rôle est de structurer et organiser les informations trouvées dans la base de données en une réponse claire et cohérente.
                
                RÈGLES IMPORTANTES :
                1. Utilise EXCLUSIVEMENT les informations fournies dans les résultats de recherche
                2. Ne jamais inventer ou ajouter d'informations
                3. Adopte un ton officiel mais accessible de la FSO
                4. Structure la réponse de manière logique et professionnelle
                5. Évite les répétitions entre les différents résultats
                6. Synthétise les informations complémentaires
                7. Organise les informations par ordre d'importance
                8. repondre en français
            
                
                STRUCTURE DE RÉPONSE :
                - Commence par une introduction brève si nécessaire
                - Organise les informations par thèmes logiques
                - Utilise des paragraphes clairs et bien structurés
                - Termine par des informations de contact si pertinent
                
                Si plusieurs résultats traitent du même sujet, combine-les intelligemment sans répétition.""",
                
                'user': """Question de l'étudiant/visiteur : {question}

                Voici TOUS les résultats trouvés dans la base de données FSO :

                {search_results}

                Merci de créer une réponse structurée et cohérente en utilisant ces informations. Organise-les de manière logique et évite les répétitions."""
            },
            
            'en': {
                'system': """You are the official virtual assistant of the Faculty of Sciences of Oujda (FSO).
                
                Your role is to structure and organize information found in the database into a clear and coherent response.
                
                IMPORTANT RULES:
                1. Use EXCLUSIVELY the information provided in search results
                2. Never invent or add information
                3. Adopt an official but accessible tone for FSO
                4. Structure the response logically and professionally
                5. Avoid repetitions between different results
                6. Synthesize complementary information
                7. Organize information by order of importance
                8. response in english
                
                RESPONSE STRUCTURE:
                - Start with a brief introduction if necessary
                - Organize information by logical themes
                - Use clear and well-structured paragraphs
                - End with contact information if relevant
                
                If multiple results address the same topic, combine them intelligently without repetition.""",
                
                'user': """Student/visitor question: {question}

                Here are ALL the results found in the FSO database:

                {search_results}

                Please create a structured and coherent response using this information. Organize it logically and avoid repetitions."""
            },
            
            'ar': {
                'system': """أنت المساعد الافتراضي الرسمي لكلية العلوم بوجدة (FSO).
                
                دورك هو تنظيم وهيكلة المعلومات الموجودة في قاعدة البيانات في إجابة واضحة ومتماسكة.
                
                قواعد مهمة:
                1. استخدم حصرياً المعلومات المقدمة في نتائج البحث
                2. لا تخترع أو تضيف معلومات أبداً
                3. اعتمد نبرة رسمية ولكن مفهومة لكلية العلوم
                4. نظم الإجابة بطريقة منطقية ومهنية
                5. تجنب التكرار بين النتائج المختلفة
                6. اجمع المعلومات المتكاملة
                7. نظم المعلومات حسب الأهمية
                
                هيكل الإجابة:
                - ابدأ بمقدمة مختصرة إذا لزم الأمر
                - نظم المعلومات حسب المواضيع المنطقية
                - استخدم فقرات واضحة ومنظمة
                - اختتم بمعلومات الاتصال إذا كان مناسباً
                
                إذا كانت عدة نتائج تتناول نفس الموضوع، اجمعها بذكاء دون تكرار.""",
                
                'user': """سؤال الطالب/الزائر: {question}

                إليك جميع النتائج الموجودة في قاعدة بيانات كلية العلوم:

                {search_results}

                يرجى إنشاء إجابة منظمة ومتماسكة باستخدام هذه المعلومات. نظمها بطريقة منطقية وتجنب التكرار."""
            },
            
            'amz': {
                'system': """Anta d amellal ufrawan unṣib n tesnawalt n tussniwin n Wujda (FSO).
                
                Tatwilt-nnek d asbedd d usbadu n talɣut i yellan deg taffa n yisefka ɣer tiririt tefrawant u teǧǧa.
                
                Ilugan ixataren:
                1. Seqdec kan talɣut i d-yettunefken deg igemmaḍ n unadi
                2. Ur d-snifl neɣ ur d-rnu ara talɣut
                3. Seqdec tasa tunṣibt maca i d-yettafehmen i tesnawalt
                4. Sbedd tiririt s tarrayt tusnakt d tsnakt
                5. Gani asniles gar igemmaḍ nniḍen
                6. Sdukkel talɣut i d-yettemsekkilen
                7. Sbedd talɣut almend n lexṣaṣ
                
                Asbadu n tririt:
                - Bdu s tezwart tawezlant ma ilaq
                - Sbedd talɣut almend n yisental ilugan
                - Seqdec tafransist tefrawant u tettusbeḍ
                - Fakk s yisalli n unrmis ma yella ifaq
                
                Ma yella aṭas n igemmaḍ i d-yemmeslayen ɣef yiwen n wennez, sdukkel-iten s tmuski ur asniles.""",
                
                'user': """Asqsi n uneɣmas/amarza: {question}

                Hatan akk igemmaḍ i d-yettwafen deg taffa n yisefka n tesnawalt:

                {search_results}

                Ttxil-k snulfu-d tiririt tettusbeḍ u teǧǧa s useqdec n telɣut-a. Sbedd-itt s tarrayt tusnakt u gani asniles."""
            }
        }
        
        self.no_results_messages = {
            'fr': "Je suis l'assistant virtuel de la Faculté des Sciences d'Oujda. Je n'ai pas trouvé d'informations spécifiques à votre question dans notre base de données. Pour plus d'informations, veuillez contacter les services administratifs de la FSO ou consulter le site web officiel.",
            'en': "I am the virtual assistant of the Faculty of Sciences of Oujda. I couldn't find specific information about your question in our database. For more information, please contact the FSO administrative services or visit the official website.",
            'ar': "أنا المساعد الافتراضي لكلية العلوم بوجدة. لم أتمكن من العثور على معلومات محددة حول سؤالك في قاعدة البيانات. للمزيد من المعلومات، يرجى الاتصال بالخدمات الإدارية للكلية أو زيارة الموقع الرسمي.",
            'amz': "Nekk d amellal ufrawan n tesnawalt n tussniwin n Wujda. Ur ufiɣ ara talɣut tazribt ɣef usqsi-nnek deg taffa n yisefka. I wugar n telɣut, nermes tanbaḍt taneggarut n tesnawalt neɣ rzu asmel unṣib."
        }

    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Appelle l'API Ollama avec optimisations GPU"""
        try:
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": self.gpu_optimized_options.copy()
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            
            start_time = datetime.now()
            logger.info(f"Appel Ollama GPU - Prompt: {len(prompt)} caractères")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60000
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                
                logger.info(f"Réponse générée en {processing_time:.2f}s")
                logger.info(f"Tokens évalués: {result.get('eval_count', 'N/A')}")
                logger.info(f"Vitesse: {result.get('eval_count', 0) / processing_time:.1f} tokens/s")
                
                return response_text
            else:
                raise Exception(f"Erreur Ollama HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de l'appel à Ollama: {str(e)}")

    def validate_comprehensive_answer(self, original_question: str, simplified_questions: List[str], 
                                    generated_answer: str, lang: str) -> Dict[str, Any]:
        """
        Enhanced validation that considers the relationship between original question,
        simplified questions, and the comprehensive answer
        """
        
        validation_prompts = {
            'fr': {
                'system': """Tu es un validateur intelligent de réponses. Ta tâche est d'évaluer si une réponse générée est pertinente et complète pour une question originale et ses sous-questions.

                RÈGLES:
                1. Évalue si la réponse traite la question originale de manière appropriée
                2. Vérifie si tous les aspects des sous-questions sont abordés
                3. Identifie les éléments manquants ou non pertinents
                4. Considère le contexte FSO (Faculté des Sciences Oujda)
                5. Retourne un score de validation détaillé

                FORMAT DE RÉPONSE:
                - valid: 1 si la réponse est globalement satisfaisante, 0 sinon
                - coverage_score: score de 0 à 1 indiquant le pourcentage de couverture
                - missing_aspects: liste des aspects non couverts
                - irrelevant_content: contenu non pertinent identifié""",

                'user': """QUESTION ORIGINALE:
                {original_question}

                SOUS-QUESTIONS:
                {simplified_questions}

                RÉPONSE GÉNÉRÉE:
                {generated_answer}

                Évalue cette réponse de manière comprehensive."""
            },
            
            'en': {
                'system': """You are an intelligent answer validator. Your task is to evaluate if a generated answer is relevant and complete for an original question and its sub-questions.

                RULES:
                1. Assess if the answer appropriately addresses the original question
                2. Verify if all aspects of sub-questions are covered
                3. Identify missing or irrelevant elements
                4. Consider FSO context (Faculty of Sciences Oujda)
                5. Return a detailed validation score

                RESPONSE FORMAT:
                - valid: 1 if answer is globally satisfactory, 0 otherwise
                - coverage_score: score from 0 to 1 indicating coverage percentage
                - missing_aspects: list of uncovered aspects
                - irrelevant_content: identified irrelevant content""",

                'user': """ORIGINAL QUESTION:
                {original_question}

                SUB-QUESTIONS:
                {simplified_questions}

                GENERATED ANSWER:
                {generated_answer}

                Evaluate this answer comprehensively."""
            }
        }
        
        try:
            prompt_config = validation_prompts.get(lang, validation_prompts['fr'])
            
            simplified_questions_text = "\n".join([f"- {q}" for q in simplified_questions])
            
            user_prompt = prompt_config['user'].format(
                original_question=original_question,
                simplified_questions=simplified_questions_text,
                generated_answer=generated_answer
            )
            
            validation_response = self._call_ollama(
                prompt=user_prompt,
                system_prompt=prompt_config['system']
            )
            
            validation_result = self._parse_validation_response(validation_response)
            
            logger.info(f"Comprehensive validation - Original: {original_question[:50]}...")
            logger.info(f"Comprehensive validation - Sub-questions: {len(simplified_questions)}")
            logger.info(f"Comprehensive validation - Result: {validation_result}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive validation: {str(e)}")
            return {
                "is_valid": False,
                "coverage_score": 0.0,
                "missing_aspects": ["validation_error"],
                "irrelevant_content": [],
                "error": str(e)
            }       
    
    def format_search_results_for_structuring(self, results: List[Dict[str, Union[str, float]]]) -> str:
        """Formate tous les résultats pour permettre au LLM de les structurer"""
        if not results:
            return "Aucun résultat trouvé."
        
        formatted_results = []
        
        for i, result in enumerate(results, 1):
            # Ensure we have the required fields
            answer = result.get('answer', 'N/A')
            confidence = result.get('confidence', result.get('score', 'N/A'))
            
            formatted_result = f"""
            ═══ RÉSULTAT {i} ═══
            Réponse: {answer}
            Score de pertinence: {confidence}"""
            
            # Optional: Add metadata if available
            if result.get('meta'):
                formatted_result += f"\nMétadonnées: {result['meta']}"
            
            # Optional: Add intent information if available
            if result.get('intent'):
                formatted_result += f"\nIntention détectée: {result['intent']}"
                
            formatted_results.append(formatted_result)
        
        # Log the formatted results for debugging
        logger.info(f"Formatted {len(formatted_results)} results for LLM processing")
        
        return "\n\n".join(formatted_results)

    def generate_comprehensive_response(self, original_question: str, question_answer_pairs: List[Dict], 
                                    all_documents: List[Dict], lang: str) -> Dict[str, Any]:
        """
        Generate a comprehensive response that synthesizes all question-answer relationships
        """
        try:
            start_time = datetime.now()
            
            comprehensive_prompts = {
                'fr': {
                    'system': """Tu es un expert en synthèse d'informations pour la Faculté des Sciences d'Oujda (FSO). 
                    Ta tâche est d'analyser plusieurs paires question-réponse et de générer une réponse comprehensive et cohérente.

                    RÈGLES IMPORTANTES:
                        1. Analyse TOUTES les questions ensemble pour comprendre le besoin complet d'information de l'utilisateur
                        2. Examine les réponses pour identifier:
                            - Les informations complémentaires qui doivent être combinées
                            - Les contradictions qui nécessitent une résolution
                            - Les lacunes qui doivent être reconnues
                        3. Structure ta réponse pour:
                            - Répondre clairement à chaque question
                            - Montrer les connexions entre questions liées
                            - Résoudre les conflits entre réponses
                            - Maintenir un flux logique
                        4. Pour les questions temporelles, indique clairement la période de chaque information
                        5. Préserve toutes les informations uniques et précieuses tout en éliminant la redondance
                        6. Si les réponses sont en conflit, indique-le et fournis toutes les perspectives
                        7. Utilise un format structuré pour les cas multi-questions complexes

                    FORMAT DE SORTIE:
                        1. Réponse synthétique qui traite tous les aspects
                        2. Indication des sources d'information
                        3. Gestion des conflits ou incertitudes si nécessaire""",

                    'user': """QUESTION ORIGINALE DE L'UTILISATEUR:
                    {original_question}

                    QUESTIONS SIMPLIFIÉES ET LEURS RÉPONSES:
                    {formatted_qa_pairs}

                    CONTEXTE GLOBAL:
                    - Total des questions: {num_questions}
                    - Total des sources: {num_sources}

                    TÂCHE: Génère une réponse comprehensive qui traite tous les aspects du besoin d'information de l'utilisateur en synthétisant toutes les informations disponibles.
                    Résous les conflits, comble les lacunes si possible, et maintiens toutes les informations précieuses tout en éliminant la redondance."""
                },
                
                'en': {
                    'system': """You are an expert information synthesizer for the Faculty of Sciences Oujda (FSO). 
                    Your task is to analyze multiple question-answer pairs and generate a comprehensive, coherent response.

                    IMPORTANT RULES:
                    1. Analyze ALL questions together to understand the user's complete information need
                    2. Cross-reference answers to identify:
                    - Complementary information that should be combined
                    - Contradictions that need resolution
                    - Gaps that need to be acknowledged
                    3. Structure your response to:
                    - Address each question clearly
                    - Show connections between related questions
                    - Resolve conflicts between answers
                    - Maintain logical flow
                    4. For temporal questions, clearly indicate the timeframe of each piece of information
                    5. Preserve all unique valuable information while eliminating redundancy
                    6. If answers conflict, indicate this and provide all perspectives
                    7. Use structured format for complex multi-question cases

                    OUTPUT FORMAT:
                    1. Synthetic response addressing all aspects
                    2. Source information indication
                    3. Conflict or uncertainty management if needed""",

                    'user': """USER'S ORIGINAL QUESTION:
                    {original_question}

                    SIMPLIFIED QUESTIONS AND THEIR ANSWERS:
                    {formatted_qa_pairs}

                    GLOBAL CONTEXT:
                    - Total questions: {num_questions}
                    - Total sources: {num_sources}

                    TASK: Generate a comprehensive response that addresses all aspects of the user's information need by synthesizing all available information. Resolve conflicts, fill gaps where possible, and maintain all valuable information while eliminating redundancy."""
                },
                
                'ar': {
                    'system': """أنت خبير في تجميع المعلومات لكلية العلوم وجدة (FSO).
                    مهمتك هي تحليل عدة أزواج من الأسئلة والأجوبة وإنتاج إجابة شاملة ومتماسكة.

                    القواعد المهمة:
                    1. حلل كل الأسئلة معاً لفهم حاجة المستخدم الكاملة للمعلومات
                    2. راجع الأجوبة لتحديد:
                    - المعلومات المكملة التي يجب دمجها
                    - التناقضات التي تحتاج حل
                    - الثغرات التي يجب الاعتراف بها
                    3. هيكل إجابتك لـ:
                    - الإجابة بوضوح على كل سؤال
                    - إظهار الروابط بين الأسئلة المترابطة
                    - حل التعارضات بين الأجوبة
                    - الحفاظ على تدفق منطقي
                    4. للأسئلة الزمنية، وضح بوضوح الإطار الزمني لكل معلومة
                    5. اعتن بكل المعلومات القيمة الفريدة مع إزالة التكرار
                    6. إذا تعارضت الأجوبة، وضح ذلك وقدم كل وجهات النظر
                    7. استخدم تنسيق منظم للحالات المتعددة الأسئلة المعقدة

                    تنسيق المخرجات:
                    1. إجابة تركيبية تتناول كل الجوانب
                    2. إشارة لمصادر المعلومات
                    3. إدارة التعارضات أو عدم اليقين إذا لزم الأمر""",

                                    'user': """السؤال الأصلي للمستخدم:
                    {original_question}

                    الأسئلة المبسطة وأجوبتها:
                    {formatted_qa_pairs}

                    السياق العام:
                    - إجمالي الأسئلة: {num_questions}
                    - إجمالي المصادر: {num_sources}

                    المهمة: أنتج إجابة شاملة تعالج كل جوانب حاجة المستخدم للمعلومات عبر تجميع كل المعلومات المتاحة. حل التعارضات، املأ الثغرات إن أمكن، واحتفظ بكل المعلومات القيمة مع إزالة التكرار."""
                }
            }
            
            prompt_config = comprehensive_prompts.get(lang, comprehensive_prompts['fr'])
            
            # Format question-answer pairs in a structured way
            formatted_pairs = self._format_comprehensive_qa_pairs(question_answer_pairs, lang)
            
            # Create the comprehensive prompt
            user_prompt = prompt_config['user'].format(
                original_question=original_question,
                formatted_qa_pairs=formatted_pairs,
                num_questions=len(question_answer_pairs),
                num_sources=len(all_documents)
            )
            
            # Generate comprehensive response
            comprehensive_response = self._call_ollama(
                prompt=user_prompt,
                system_prompt=prompt_config['system']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate comprehensive confidence
            confidence = self._calculate_comprehensive_confidence(question_answer_pairs, all_documents)
            
            # Analyze question coverage
            question_coverage = self._analyze_question_coverage(question_answer_pairs, comprehensive_response)
            
            logger.info(f"Comprehensive response generated with confidence: {confidence}")
            
            return {
                'response': comprehensive_response,
                'confidence': confidence,
                'sources_used': len(all_documents),
                'questions_addressed': len(question_answer_pairs),
                'processing_time': processing_time,
                'question_coverage': question_coverage,
                'scope': 'comprehensive_fso'
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive response: {str(e)}")
            return {
                'response': self.no_results_messages.get(lang, 'Erreur lors du traitement.'),
                'confidence': 0.0,
                'sources_used': 0,
                'questions_addressed': 0,
                'processing_time': 0,
                'error': str(e),
                'scope': 'error'
            }      
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
            """Calcule un score de confiance basé sur les résultats"""
            if not results:
                return 0.0
            
            # Normaliser les scores entre 0 et 1 si nécessaire
            scores = []
            for r in results:
                score = r.get('score', 0)
                # Si les scores sont élevés (comme dans votre exemple 23.21), normalisez-les
                if score > 1.0:
                    score = score / 100  # Ajustez ce facteur selon votre échelle de score
                scores.append(score)
            
            avg_score = sum(scores) / len(scores)
            
            # pour plusieurs sources
            source_bonus = min(len(results) * 0.05, 0.2)
            
            # si les scores sont élevés
            high_score_bonus = 0.1 if avg_score > 0.8 else 0.0
            
            final_confidence = min(avg_score + source_bonus + high_score_bonus, 1.0)
            logger.info(f"Confidence = {round(final_confidence, 2)}")
            return round(final_confidence, 2)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Récupère les informations sur le modèle utilisé"""
        try:
            response = requests.get(f"{self.base_url}/api/show", 
                                  json={"name": self.model_name}, 
                                  timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Erreur HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de performance"""
        try:
            stats = {
                "model": self.model_name,
                "gpu_config": {
                    "cuda_device": os.environ.get('CUDA_VISIBLE_DEVICES'),
                    "gpu_layers": os.environ.get('OLLAMA_GPU_LAYERS'),
                    "parallel_requests": os.environ.get('OLLAMA_NUM_PARALLEL'),
                    "max_models": os.environ.get('OLLAMA_MAX_LOADED_MODELS')
                },
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
                    "memory_available": f"{psutil.virtual_memory().available / (1024**3):.1f}GB"
                }
            }
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    stats["gpu_info"] = [
                        {
                            "name": gpu.name,
                            "memory_total": f"{gpu.memoryTotal}MB",
                            "memory_used": f"{gpu.memoryUsed}MB",
                            "load": f"{gpu.load*100:.1f}%"
                        } for gpu in gpus
                    ]
            except:
                pass
            
            return stats
        except Exception as e:
            return {"error": str(e)}

    def _format_context_for_llama(self, context: Dict[str, Any]) -> str:
        """Formate le contexte de manière optimale pour Llama3:8b"""
        try:
            formatted_parts = []
            
            for key, value in context.items():
                if isinstance(value, (str, int, float)):
                    formatted_parts.append(f"- {key}: {value}")
                elif isinstance(value, list):
                    if len(value) <= 5:
                        formatted_parts.append(f"- {key}: {', '.join(map(str, value))}")
                    else:
                        formatted_parts.append(f"- {key}: {', '.join(map(str, value[:5]))} (et {len(value)-5} autres)")
                elif isinstance(value, dict):
                    # Flatten nested dicts to avoid complexity
                    sub_items = []
                    for sub_key, sub_value in list(value.items())[:3]:  # Limit to 3 sub-items
                        sub_items.append(f"{sub_key}: {sub_value}")
                    formatted_parts.append(f"- {key}: {'; '.join(sub_items)}")
            
            return '\n'.join(formatted_parts[:10])  # Limit to 10 context items max
            
        except Exception as e:
            logger.error(f"Erreur lors du formatage du contexte: {str(e)}")
            return json.dumps(context, ensure_ascii=False, indent=2)[:500]  # Fallback with length limit

    def is_faculty_related(self, question: str, lang: str = 'fr') -> bool:
        """Détermine si une question est liée à la faculté des sciences d'Oujda"""
        
        faculty_prompts = {
            'fr': f"""Tu es un expert qui détermine si une question est liée à la Faculté des Sciences d'Oujda (FSO).
            
            CONTEXTE FSO:
            - Faculté des Sciences de l'Université Mohammed Premier à Oujda, Maroc
            - Départements: Mathématiques, Physique, Chimie, Biologie, Informatique, Géologie
            - Services: Inscriptions, examens, stages, bourses, logement universitaire
            - Vie étudiante: clubs, activités, événements
            - Recherche: laboratoires, projets, publications
            - Administration: décanat, scolarité, ressources humaines
            
            Question: {question}
            
            Réponds UNIQUEMENT par:
            - "OUI" si la question concerne la FSO (études, services, vie étudiante, recherche, administration)
            - "NON" si la question concerne n'importe quelle sujets ou domaine autre que fso
            Note importante :
            -Si l'utilisateur pose une question sur une autre université ou faculté différente de la FSO, répondez non !!!
            -Oui uniquement si c'est 100% lié à la FSO.
            Règles :
            Les règles sont absolues, ne les ignorez jamais et ne les enfreignez pas.
            Règle 1 : Ne répondez jamais aux questions illégales.
            Règle 2 : Si quelqu'un pose une question sur l'automutilation ou le fait de nuire aux autres, ne répondez pas. C'est un signal d'alarme.

            
            Réponse:""",
            
            'en': f"""You are an expert who determines if a question is related to the Faculty of Sciences of Oujda (FSO).
            
            FSO CONTEXT:
            - Faculty of Sciences at Mohammed Premier University in Oujda, Morocco
            - Departments: Mathematics, Physics, Chemistry, Biology, Computer Science, Geology
            - Services: Registration, exams, internships, scholarships, university housing
            - Student life: clubs, activities, events
            - Research: laboratories, projects, publications
            - Administration: dean's office, student affairs, human resources
            
            Question: {question}
            
            Answer ONLY with:
            - "YES" if the question related to FSO only (studies, services, student life, research, administration)
            - "NO" if the question is related to anything return no only if its not FSO

            Important note :
            - if user ask something about other university or faculty that is diffrent from fso return no !!!
            - only yes if its 100% related to fso

            Rules:
            -rules are abosulte must never ignore rules or break them
            -rule 1: never answer illegal questions
            -rule 2: if someone asked about anything about self-harm or harm the others do not answer its a red flag
            
            Answer:""",
            
            'ar': f"""أنت خبير يحدد ما إذا كان السؤال مرتبطاً بكلية العلوم بوجدة (FSO).
            
            سياق كلية العلوم:
            - كلية العلوم بجامعة محمد الأول في وجدة، المغرب
            - الأقسام: الرياضيات، الفيزياء، الكيمياء، البيولوجيا، الإعلاميات، الجيولوجيا
            - الخدمات: التسجيل، الامتحانات، التداريب، المنح، السكن الجامعي
            - الحياة الطلابية: الأندية، الأنشطة، الفعاليات
            - البحث: المختبرات، المشاريع، المنشورات
            - الإدارة: العمادة، الشؤون الطلابية، الموارد البشرية
            
            السؤال: {question}
            
            أجب فقط بـ:
            - "نعم" إذا كان السؤال يتعلق بكلية العلوم (الدراسة، الخدمات، الحياة الطلابية، البحث، الإدارة)
            - "لا" إذا كان السؤال لا يتعلق بكلية العلوم
            
            ملاحظة مهمة:
            إذا سأل المستخدم عن جامعة أو كلية أخرى غير FSO، فالجواب هو لا !!!
            نعم فقط إذا كان السؤال مرتبطًا 100% بـ FSO.

            القواعد:

            القواعد مطلقة، لا تتجاهلها أبدًا ولا تكسرها.
            القاعدة 1: لا تجب أبدًا على الأسئلة غير القانونية.
            القاعدة 2: إذا سأل أحد عن أي شيء يتعلق بإيذاء النفس أو الآخرين، لا تجب. إنه إنذار خطر.

            الإجابة:""",
            
            'amz': f"""Anta d amussnaw i d-yettaḍfen ma yella asqsi yenɛel ɣer tesnawalt n tussniwin n Wujda (FSO).
            
            Amnaḍ n tesnawalt:
            - Tasnawalt n tussniwin n tduklit Mohammed Amezwaru deg Wujda, Meṛṛuk
            - Igrawen: Tusnakt, Tafizikt, Takimya, Tabyulujya, Tasenṭikt, Tajyulujya
            - Tanbaḍt: Askalas, imtihanan, asɣar, tikriyin, amagger ajamɛi
            - Tudert n yineɣmasen: ikluban, tigawin, tidyanin
            - Unadi: tinaɣin, isenfaren, tizragin
            - Tanbaḍt: lɛamada, uguren n yineɣmasen, yiɣbula n wemdan
            
            Asqsi: {question}
            
            Rrar kan s:
            - "IH" ma yella asqsi yenɛel ɣer FSO (tizrawin, tanbaḍt, tudert n yineɣmasen, unadi, tanbaḍt)
            - "UHU" ma yella asqsi ur yenɛel ara ɣer FSO
            ⵜⵓⵙⵙⵏⴰ ⵎⵀⵉⵎⵎⴰ:
            ⵎⴰ ⵢⵙⴰ ⵍⵃⴷⵎ ⵙ ⵜⵎⵙⵙⵏⴰ ⵏ ⵜⵓⵏⵉⴼⵔⵙⵉⵜ ⵏⵉⵙⵜ ⵓ ⵜⵓⵙⵍⴰ ⵜⴰⵎⵙⵙⵓⵜⵏ ⵖⵔ FSO, ⵙⵙⵓⵜⵉⴷ ⵓⵀⵓ !!!
            ⵢⴰⵀ ⵙⵉⴼ ⵜⵜⵎⵙⵙⵉⵍⴷ 100% ⵙ FSO
                        Tiririt:
            ⵍⵇⵡⴰⵏⵉⵏ:

            ⵍⵇⵡⴰⵏⵉⵏ ⵏⵖⵏ ⵓⵍⴰ ⵜⵙⵏⵖⵍⵉⵜ ⴰⵎⴰ ⵜⵜⴰⵖⵍⵉⵜ.
            ⵍⵇⴰⵏⵓⵏ 1: ⵎⴰⵀⵉ ⵜⵙⵙⵓⵜⵉⴷ ⵙ ⵜⵉⵎⵙⵉⵍⴰ ⵜⵉⵍⵉⴳⴰⵍ.
            ⵍⵇⴰⵏⵓⵏ 2: ⵎⴰ ⵢⵙⴰ ⵍⵃⴷⵎ ⵙ ⵓⵎⴰ ⵢⵉⵙⵙⵏ ⵏ ⵓⵎⵙⵙⵉ ⵏⵏⵙ ⵓ ⵏ ⵍⵃⴷⵎ, ⵎⴰ ⵜⵙⵙⵓⵜⵉⴷ. ⵜⵜⵓⵔⴷⵉⵜ ⵜⴰⵣⴳⵯⵔⴰ.""" 
        }
        
        try:
            prompt = faculty_prompts.get(lang, faculty_prompts[lang])
            response = self._call_ollama(prompt=prompt)
            
            # Analyser la réponse
            response_lower = response.lower().strip()
            
            # Mots-clés positifs selon la langue
            positive_keywords = {
                'fr': ['oui', 'yes', 'si', 'correct', 'vrai'],
                'en': ['yes', 'oui', 'correct', 'true'],
                'ar': ['نعم', 'yes', 'oui', 'صحيح'],
                'amz': ['ih', 'yes', 'oui', 'akken']
            }
            
            is_related = any(keyword in response_lower for keyword in positive_keywords.get(lang, positive_keywords[lang]))
            
            logger.info(f"is_related return : {is_related}")
            return is_related
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de pertinence: {str(e)}")
            return False

    def generate_faculty_response(self, question: str, lang: str = 'fr') -> Dict[str, Any]:
        """Génère une réponse basée sur le modèle fine-tuné avec vos données FSO"""
        
        # Prompts optimisés pour votre modèle fine-tuné
        finetuned_prompts = {
            'fr': f"""Tu es l'assistant virtuel officiel de la Faculté des Sciences d'Oujda (FSO) de l'Université Mohammed Premier.

            Tu as été entraîné spécifiquement sur les données de la FSO. Utilise UNIQUEMENT tes connaissances sur la FSO pour répondre.

            STRICTEMENT FSO SEULEMENT:
            - Faculté des Sciences d'Oujda, Université Mohammed Premier, Maroc
            - Évite toute confusion avec d'autres facultés (Lettres, Économie, etc.)
            - Réponds uniquement sur ce qui concerne la FSO

            Question: {question}

            Réponse précise et factuelle basée sur tes données d'entraînement FSO:""",

            'en': f"""You are the official virtual assistant of the Faculty of Sciences of Oujda (FSO), 
            Mohammed Premier University.

            You have been specifically trained on FSO data. Use ONLY your FSO knowledge to respond.

            STRICTLY FSO ONLY:
            - Faculty of Sciences of Oujda, Mohammed Premier University, Morocco  
            - Avoid confusion with other faculties (Letters, Economics, etc.)
            - Answer only about FSO-related matters
            - response in english
            Question: {question}

            Precise and factual response based on your FSO training data:""",

            'ar': f"""أنت المساعد الافتراضي الرسمي لكلية العلوم بوجدة (FSO) بجامعة محمد الأول.

            تم تدريبك خصيصاً على بيانات كلية العلوم. استخدم فقط معرفتك بكلية العلوم للإجابة.

            كلية العلوم فقط:
            - كلية العلوم بوجدة، جامعة محمد الأول، المغرب
            - تجنب الخلط مع كليات أخرى (الآداب، الاقتصاد، إلخ)
            - أجب فقط عما يخص كلية العلوم
            - الرد باللغة العربية

            السؤال: {question}

            إجابة دقيقة وواقعية مبنية على بيانات تدريبك لكلية العلوم:""",

            'amz': f"""Anta d amellal ufrawan unṣib n tesnawalt n tussniwin n Wujda (FSO) n tduklit Mohammed Amezwaru.

            Tettwaselmadeḍ s talɣa tusligt ɣef yisefka n FSO. Seqdec kan tamusni-nnek n FSO i tiririt.

            FSO KAN:
            - Tasnawalt n tussniwin n Wujda, tduklit Mohammed Amezwaru, Meṛṛuk
            - Ur ttexleḍ ara d tesnawalt-nniḍen (Adlis, Tadamsa, atg.)
            - Rrar kan ɣef wayen yeɛnan FSO

            Asqsi: {question}

            Tiririt d tameɣtut d tameɣnut s talɣa n yisefka n useɣɣef-nnek FSO:"""
        }
        
        try:
            prompt = finetuned_prompts.get(lang, finetuned_prompts['fr'])
            response = self._call_ollama(prompt=prompt)
            
            return {
                'response': response,
                'confidence': 0.9,
                'source': 'finetuned_model',
                'lang': lang
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse fine-tunée: {str(e)}")
            return {
                'response': self.no_results_messages.get(lang, self.no_results_messages['fr']),
                'confidence': 0.0,
                'source': 'error',
                'lang': lang,
                'error': str(e)
            }

    def format_for_database(self, question: str, response: str, lang: str = 'fr') -> Dict[str, Any]:
        """Formate la question et la réponse pour l'insertion dans la base de données"""
        
        return {
            'question': {lang: [question]},
            'reponse': {lang: [response]},
            'meta': {lang: ['Généré par LLM - Connaissances générales FSO']}
        }

    def process_serp_to_response(self, question: str, serp_data: str, lang: str, 
                    store_to_file: bool = True, filename: str = "test.json") -> Dict[str, Any]:        
        """
            1. Analyse des résultats SERP :
                - Convertir le texte SERP en minuscules.
                - Compter les occurrences d’indicateurs FSO.

            2. Vérification du score FSO :
                - Si indicators_score < 2 → considérer les résultats comme insuffisamment pertinents.
                - Action : utiliser directement generate_faculty_response (fallback).

            3. Préparation du prompt système :
                - Répondre uniquement sur la FSO.
                - Ignorer toute autre faculté ou établissement.
                - Maintenir la question utilisateur intacte.
                - Produire un JSON structuré avec knowledge_entry.

            4. Filtrage des données SERP :
                - Appeler _filter_fso_content pour nettoyer les résultats.
                - Supprimer tout contenu hors FSO avant envoi au LLM.

            5. Construction du user_prompt :
            - Inclure :
                - Question originale de l’utilisateur.
                - Résultats SERP filtrés.
                - Instructions pour format JSON attendu.

            6. Appel API LLM :
                - Envoyer le system_prompt et le user_prompt à l’API LLM (GPU activé).
                - Enregistrer le temps de réponse.

            7. Extraction de la réponse LLM :
                - Extraire le JSON de la réponse.
                - Charger en objet Python (json.loads).

            8. Validation du JSON :
                - Si is_fso_relevant est false → fallback vers generate_faculty_response.
                - Vérifier que la question FR correspond bien à celle posée.

            9. Stockage :
            - Si store_to_file=True et knowledge_entry existe → sauvegarder le JSON via store_to_json_file.

            10. Retour des données finales :
                - Retourner un dictionnaire contenant :
                    {
                        - display : réponse textuelle pour l’utilisateur.
                        - storage : données structurées.
                        - confidence : niveau de confiance.
                        - file_path : chemin du fichier si stocké.
                        - processing_time : temps total de traitement.
                        - indicators : score et détails.
                        - rejected_content : contenu supprimé.
                    }
            11. Gestion des erreurs :
                - Si exception levée → fallback vers generate_faculty_response --> reponse par LLM lui meme.
        """

        try:
            fso_indicators = [
                "faculté des sciences", "fso", "sciences oujda", 
                "mohammed premier", "ump.ac.ma", "fso.ump.ma"
            ]
            
            serp_text = str(serp_data).lower()
            fso_score = sum(1 for indicator in fso_indicators if indicator in serp_text)
            
            if fso_score < 2:
                logger.warning("SERP data contains little FSO content, using fine-tuned model instead")
                return self.generate_faculty_response(question, lang)
            
            
            system_prompt = (
                "TASK: Process SERP data for Faculty of Sciences Oujda (FSO) ONLY\n"
                "CRITICAL: Filter out any content from other faculties (Letters, Economics, EST, etc.)\n"
                "1. Use EXACT client question - DO NOT rephrase\n"
                "2. Extract ONLY FSO-relevant facts from SERP data\n"
                "3. Reject any information about other faculties\n"
                "4. If no FSO-specific info, indicate clearly\n"
                "Language: " + str(lang) + "\n"
            )
            
            
            json_template = """{
                "user_response": "Answer based ONLY on FSO-related SERP data",
                "validation": {
                    "is_fso_relevant": true,
                    "rejected_content": ["list any non-FSO content found"],
                    "confidence_fso": 0.8
                },
                "knowledge_entry": {
                    "intent": "fso_specific_info",
                    "question": {
                        "fr": [
                            "EXACT ORIGINAL CLIENT QUESTION FOR FSO",
                            "FSO-focused variation of question",
                            "another FSO-specific variation"
                        ],
                        "en": [...],
                        "ar": [...], 
                        "amz": [...]
                    },
                    "reponse": {
                        "fr": [
                            "FSO-specific answer from SERP data",
                            "variation focusing on FSO only",
                            "another FSO-focused answer"
                        ],
                        "en": [...],
                        "ar": [...],
                        "amz": [...]
                    },
                    "meta": {
                        "fr": ["FSO source links only"],
                        "en": ["FSO source links only"],
                        "ar": ["FSO source links only"],
                        "amz": ["FSO source links only"]
                    }
                },
                "confidence": 0.8
            }"""
            
            
            formatted_serp_data = self._filter_fso_content(serp_data)
            
            user_prompt = (
                "ORIGINAL CLIENT QUESTION FOR FSO: '" + str(question) + "'\n\n" +
                "IMPORTANT: This question is specifically about Faculty of Sciences Oujda (FSO)\n" +
                "REJECT any content about other faculties (Letters, Economics, EST, etc.)\n\n" +
                "FSO-FILTERED SERP DATA:\n" + formatted_serp_data + "\n\n" +
                "INSTRUCTIONS:\n" +
                "1. Use exact original question: '" + str(question) + "'\n" +
                "2. Extract ONLY FSO-relevant information\n" +
                "3. Mark is_fso_relevant as false if no FSO content found\n" +
                "4. List any rejected non-FSO content\n\n" +
                "OUTPUT FORMAT (JSON):\n" + 
                json_template + "\n\n" +
                "Generate the FSO-validated JSON response:"
            )
            
            
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": self.gpu_optimized_options.copy()
            }
            
            start_time = datetime.now()
            logger.info(f"Processing FSO-validated SERP for: {question[:50]}...")
            
            response = requests.post(
                self.base_url + "/api/generate",
                json=payload,
                timeout=60000
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get('response', '').strip()
                
                
                json_start = llm_output.find('{')
                json_end = llm_output.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = llm_output[json_start:json_end]
                    processed = json.loads(json_str)
                    
                    
                    validation = processed.get('validation', {})
                    if not validation.get('is_fso_relevant', False):
                        logger.warning("SERP content not FSO-relevant, falling back to fine-tuned model")
                        return self.generate_faculty_response(question, lang)
                    
                    knowledge_entry = processed.get('knowledge_entry', {})
                    
                    
                    questions = knowledge_entry.get('question', {})
                    if questions.get('fr') and isinstance(questions['fr'], list):
                        if questions['fr'][0] != question:
                            questions['fr'][0] = question
                    
                    
                    storage_success = False
                    if store_to_file and knowledge_entry:
                        storage_success = self.store_to_json_file(knowledge_entry, filename)
                    
                    return {
                        'display': processed.get('user_response', 'No FSO response generated'),
                        'storage': knowledge_entry,
                        'confidence': processed.get('confidence', 0.7),
                        'stored_to_file': storage_success,
                        'file_path': filename if storage_success else None,
                        'original_question': question,
                        'processing_time': processing_time,
                        'fso_validated': True,
                        'rejected_content': validation.get('rejected_content', [])
                    }
                else:
                    raise ValueError("No valid JSON in LLM response")
                    
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error in FSO SERP processing: {str(e)}")
            logger.info("Falling back to fine-tuned model due to SERP processing error")
            return self.generate_faculty_response(question, lang)

    def enhance_response_with_context(self, response: str, context: Dict[str, Any], lang: str = 'fr') -> str:
        """
        Améliore la réponse en ajoutant du contexte pertinent sans modifier les faits,
        sans répéter ce qui est déjà présent, et en gardant la réponse claire et structurée.
        Fonctionne pour tout sujet avec focus sur FSO.
        """
        try:
            base_instructions = {
                'fr': f"""
                    Tu es un assistant spécialisé dans les informations sur la Faculté des Sciences d'Oujda (FSO).

                    Règles STRICTES :
                    - Améliore UNIQUEMENT la réponse en ajoutant des informations NOUVELLES du contexte
                    - INTERDICTION ABSOLUE de répéter des informations déjà présentes dans la réponse
                    - INTERDICTION de créer des doublons, triplications ou répétitions de blocs entiers
                    - Focus PRIORITAIRE sur "FSO", "Faculté des Sciences Oujda", "Université Mohammed Premier"
                    - Évite les références à "CAP-FSO" sauf si directement pertinent à la question
                    - Supprime tous les doublons et répétitions avant de répondre
                    - Garde seulement les informations qui répondent EXACTEMENT à la question posée
                    - Structure claire : UNE SEULE mention par information/personne/détail
                    - Réponse finale naturelle, fluide et SANS RÉPÉTITION
                    - reponds en français
                    EXEMPLE DE CE QUI EST INTERDIT :
                    - Répéter "Doyen: Professeur El Bekkaye MAAROUF" plusieurs fois
                    - Dupliquer les coordonnées de contact
                    - Tripler les mêmes blocs d'informations

                    Réponse actuelle :
                    {response}

                    Contexte disponible :
                    {json.dumps(context, ensure_ascii=False, indent=2)}

                    CRITIQUE : Analyse d'abord la réponse actuelle pour identifier les répétitions, puis donne UNIQUEMENT la version finale améliorée, nettoyée de TOUS les doublons.
                    """,
                
                'en': f"""
                    You are an assistant specialized in information about the Faculty of Sciences of Oujda (FSO).

                    STRICT Rules:
                    - Improve ONLY the response by adding NEW information from the context
                    - ABSOLUTE PROHIBITION of repeating information already in the response
                    - PRIORITY focus on "FSO", "Faculty of Sciences Oujda", "Mohammed Premier University"
                    - Avoid references to "CAP-FSO" unless directly relevant to the question
                    - Remove all duplicates and repetitions
                    - Keep only information that EXACTLY answers the asked question
                    - Clear structure: single mention per piece of information
                    - Final response natural and fluent
                    - reponse in english
                    Current response:
                    {response}

                    Available context:
                    {json.dumps(context, ensure_ascii=False, indent=2)}

                    IMPORTANT: Give ONLY the final improved version, no duplicates, no repetitions, no commentary.
                """,
                
                'ar': f"""
                أنت مساعد متخصص في معلومات كلية العلوم بوجدة (FSO).

                قواعد صارمة:
                - حسّن الإجابة فقط بإضافة معلومات جديدة من السياق
                - منع مطلق لتكرار المعلومات الموجودة في الإجابة
                - تركيز أولوي على "FSO"، "كلية العلوم وجدة"، "جامعة محمد الأول"
                - تجنب الإشارة إلى "CAP-FSO" إلا إذا كانت متعلقة مباشرة بالسؤال
                - احذف جميع التكرارات والمضاعفات
                - احتفظ بالمعلومات التي تجيب بالضبط على السؤال المطروح
                - هيكل واضح: ذكر واحد لكل معلومة
                - إجابة نهائية طبيعية وسلسة
                - الرد باللغة العربية

                الإجابة الحالية:
                {response}

                السياق المتاح:
                {json.dumps(context, ensure_ascii=False, indent=2)}

                مهم: أعطِ النسخة النهائية المحسنة فقط، بدون تكرار، بدون تعليقات.
                """,
                
                'amz': f"""
                Anta d amellal i yeẓran ɣef Fakulté des Sciences n Wejda (FSO).

                Ilugan iǧehden:
                - Seǧhed kan tiririt s useɣti n yisallen imaynuten seg umnaḍ
                - Agdel aṭas n useɣti n yisallen i d-yellan yakan deg tiririt
                - Tazwart tamezwarut i "FSO", "Fakulté des Sciences Oujda", "Tasdawit Mohammed Premier"
                - Zgel tinmal i "CAP-FSO" ala ma yella yeɛnan srid i usteqsi
                - Kkes akk inekta d useɣti
                - Ǧǧ kan isallen i d-yettarran s tṣaḥit i usteqsi
                - Askil afsus: yiwet n tenna i yal isalan
                - Tiririt taneggaru tagamant d tafessast

                Tiririt tura:
                {response}

                Amnaḍ i yellan:
                {json.dumps(context, ensure_ascii=False, indent=2)}

                Muhim: Efk kan lqem aneggaru yettwaseǧden, ur teɣreḍ ara, ur tečč ara awalen.
                """
            }
            
            prompt = base_instructions.get(lang, base_instructions['fr'])
            
            enhanced_response = self._call_ollama(prompt=prompt)
            enhanced_response = self._remove_duplicates(enhanced_response)
            
            return enhanced_response
        
        except Exception as e:
            logger.error(f"Erreur lors de l'amélioration avec contexte: {str(e)}")
            return response

    def _remove_duplicates(self, text: str) -> str:
        """
        Supprime les phrases et blocs dupliqués dans le texte
        """
        try:
            text = self._remove_block_duplicates(text)
            sentences = text.split('.')
            unique_sentences = []
            seen = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence.lower() not in seen:
                    seen.add(sentence.lower())
                    unique_sentences.append(sentence)
            
            return '. '.join(unique_sentences).strip()
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des doublons: {str(e)}")
            return text

    def _remove_block_duplicates(self, text: str) -> str:
        """
        Supprime les blocs de texte identiques répétés
        """
        try:
            paragraphs = text.split('\n\n')
            unique_paragraphs = []
            seen_paragraphs = set()
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph:
                    normalized = ' '.join(paragraph.lower().split())
                    if normalized not in seen_paragraphs:
                        seen_paragraphs.add(normalized)
                        unique_paragraphs.append(paragraph)
            
            return '\n\n'.join(unique_paragraphs)
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des blocs dupliqués: {str(e)}")
            return text

    def _filter_context_for_fso(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filtre le contexte pour se concentrer sur FSO 
        """
        try:
            filtered_context = {}
            
            fso_keywords = [
                'faculté des sciences',
                'faculty of sciences',
                'fso',
                'oujda',
                'université mohammed premier',
                'mohammed first university',
                'ump'
            ]
            
            
            avoid_keywords = [
                'cap-fso',
                'cap fso',
                'commission académique'
            ]
            
            for key, value in context.items():
                if isinstance(value, str):
                    if any(keyword in value.lower() for keyword in fso_keywords):
                        if not any(avoid in value.lower() for avoid in avoid_keywords):
                            filtered_context[key] = value
                        elif any(fso_kw in value.lower() for fso_kw in fso_keywords[:3]):
                            filtered_context[key] = value
                else:
                    filtered_context[key] = value
            
            return filtered_context if filtered_context else context
            
        except Exception as e:
            logger.error(f"Erreur lors du filtrage du contexte: {str(e)}")
            return context

    def build_enhanced_serp_query(self, question: str, lang: str = 'fr') -> str:
        """Construit une requête SERP améliorée pour éviter les autres facultés"""
        
        
        fso_sites = [
            "site:fso.ump.ma",
            "site:ump.ac.ma/fso", 
            "site:sciences.ump.ac.ma"
        ]
        
        
        exclude_sites = [
            "-site:flsh.ump.ac.ma",     
            "-site:est.ump.ac.ma",     
            "-site:encg.ump.ac.ma",     
            "-site:fsjes.ump.ac.ma",   
            "-inurl:lettres",
            "-inurl:economie", 
            "-inurl:droit",
            "-inurl:est"
        ]
        
        
        fso_keywords = {
            'fr': ['"faculté sciences"', '"FSO"', '"sciences oujda"'],
            'en': ['"faculty sciences"', '"FSO"', '"sciences oujda"'],
            'ar': ['"كلية العلوم"', '"العلوم وجدة"'],
            'amz': ['"tasnawalt tussniwin"']
        }
        
        
        sites_part = " OR ".join(fso_sites)
        exclude_part = " ".join(exclude_sites)
        keywords = " ".join(fso_keywords.get(lang, fso_keywords['fr']))
        
        
        enhanced_query = f"({sites_part}) {keywords} {question} {exclude_part}"
        
        logger.info(f"Enhanced SERP query: {enhanced_query}")
        return enhanced_query

    def _filter_fso_content(self, serp_data: str) -> str:
        """Filtre le contenu SERP pour garder seulement les données FSO"""
        
        if isinstance(serp_data, dict):
            serp_data = str(serp_data)
        
        
        fso_positive = [
            "faculté des sciences", "fso", "sciences oujda", 
            "ump.ac.ma", "fso.ump.ma", "mohammed premier"
        ]
        
        fso_negative = [
            "faculté des lettres", "flsh", "économie", "fsjes",
            "est oujda", "encg", "droit", "lettres"
        ]
        
        lines = serp_data.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_lower = line.lower()
            
            
            has_negative = any(neg in line_lower for neg in fso_negative)
            if has_negative:
                continue
                
            
            has_positive = any(pos in line_lower for pos in fso_positive)
            if has_positive or len(line.strip()) < 50:  # Lignes courtes probablement neutres
                filtered_lines.append(line)
        
        filtered_content = '\n'.join(filtered_lines)
        
        
        if len(filtered_content) > 2000:
            filtered_content = filtered_content[:2000] + "... [filtered and truncated]"
        
        logger.info(f"Filtered SERP content: {len(serp_data)} -> {len(filtered_content)} chars")
        return filtered_content

    def get_hybrid_response(self, question: str, lang: str = 'fr') -> Dict[str, Any]:
        """Méthode hybride: modèle fine-tuné d'abord, SERP en fallback"""
        
        
        logger.info("Trying fine-tuned model first...")
        finetuned_response = self.generate_faculty_response(question, lang)
        
        
        response_text = finetuned_response.get('response', '').lower()
        
        
        weak_indicators = [
            "je ne sais pas", "don't know", "لا أعرف", "ur ẓriɣ ara",
            "pas d'information", "no information", "لا توجد معلومات",
            "désolé", "sorry", "آسف", "suref"
        ]
        
        has_weak_response = any(indicator in response_text for indicator in weak_indicators)
        is_too_short = len(response_text.strip()) < 50
        
        
        if has_weak_response or is_too_short:
            logger.info("Fine-tuned response weak, trying SERP enhancement...")
            
            try:
                enhanced_query = self.build_enhanced_serp_query(question, lang)
                serp_data = self.search_web(enhanced_query)
                
                if serp_data:
                    serp_response = self.process_serp_to_response(question, serp_data, lang)
                    
                    
                    if serp_response.get('confidence', 0) > 0.5:
                        return {
                            **serp_response,
                            'source': 'hybrid_finetuned_serp',
                            'fallback_used': True
                        }
            
            except Exception as e:
                logger.warning(f"SERP fallback failed: {str(e)}")
        
        
        return {
            **finetuned_response,
            'fallback_used': False
        }

    def simplify_question(self, question: str, lang: str = 'fr', date: datetime = None) -> list:
        """
        Simplifie une question complexe en extrayant les questions principales.
        Si la question contient plusieurs sous-questions non relatives, les sépare.
        Détermine si chaque question est statique ou dynamique selon des critères temporels.
        
        Args:
            question: Question à simplifier
            lang: Langue ('fr', 'en', 'ar', 'amz')
            date: Date de référence des connaissances (défaut: datetime.now())
        
        Returns: List of dict with 'question', 'type', and 'reason' keys
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if date is None:
            date = datetime.now()
        
        knowledge_base_date = date.strftime("%Y-%m-%d")
        
        simplification_prompts = {
            'fr': f"""Tu es un expert en analyse de questions qui simplifie les questions complexes.

            TEMPS SYSTÈME: {current_time}

            TÂCHE: Analyse cette question et détermine s'il s'agit d'une question unique complexe ou de plusieurs questions distinctes.

            RÈGLES:
            1. Si c'est UNE SEULE question complexe avec des détails supplémentaires sur le MÊME sujet → Simplifie en une question courte
            2. Si ce sont PLUSIEURS questions distinctes sur des sujets DIFFÉRENTS → Sépare chaque question et simplifie-les

            EXEMPLES:

            Question complexe unique:
            "Je voudrais savoir qui occupe actuellement le poste de Doyen de la Faculté des Sciences à l'Université Mohammed Premier d'Oujda. Pourriez-vous me fournir son nom complet, son parcours académique et professionnel, la date de prise de fonction, ainsi qu'une description de ses responsabilités, réalisations et sa vision pour la faculté?"
            → RÉSULTAT: ["Qui est le doyen de la Faculté des Sciences à l'Université Mohammed Premier d'Oujda ?"]

            Plusieurs questions distinctes:
            "Qui est le doyen de la Faculté des Sciences d'Oujda ? Aussi, quels sont les derniers résultats de l'équipe de football de Barcelona ? Et comment faire un gâteau au chocolat ?"
            → RÉSULTAT: ["Qui est le doyen de la Faculté des Sciences d'Oujda ?", "Quels sont les derniers résultats de Barcelona ?", "Comment faire un gâteau au chocolat ?"]

            INSTRUCTIONS:
            1. Lis attentivement la question
            2. Identifie s'il y a UN sujet principal ou PLUSIEURS sujets distincts
            3. Si UN sujet → Simplifie en gardant l'essentiel
            4. Si PLUSIEURS sujets → Sépare et simplifie chaque question
            5. Garde seulement les informations essentielles dans chaque question simplifiée
            6. reponds en français
            QUESTION À ANALYSER: "{question}"

            Format de réponse OBLIGATOIRE:
            ANALYSE: [Une seule question complexe / Plusieurs questions distinctes]
            RÉSULTAT: ["question simplifiée 1", "question simplifiée 2", ...]

            ANALYSE:""",

            'en': f"""You are an expert in question analysis who simplifies complex questions.

                SYSTEM TIME: {current_time}

                TASK: Analyze this question and determine if it's a single complex question or multiple distinct questions.

                RULES:
                1. If it's ONE complex question with additional details about the SAME topic → Simplify into one short question
                2. If it's MULTIPLE distinct questions about DIFFERENT topics → Separate each question and simplify them

                EXAMPLES:

                Single complex question:
                "I would like to know who is currently serving as the Dean of the Faculty of Sciences at Mohammed First University in Oujda. Could you please provide their full name, academic and professional background, the date they assumed office, as well as a description of their responsibilities, achievements, and their vision for the faculty?"
                → RESULT: ["Who is the Dean of the Faculty of Sciences at Mohammed First University in Oujda?"]

                Multiple distinct questions:
                "Who is the dean of the Faculty of Sciences in Oujda? Also, what are the latest Barcelona football team results? And how to make a chocolate cake?"
                → RESULT: ["Who is the dean of the Faculty of Sciences in Oujda?", "What are Barcelona's latest results?", "How to make a chocolate cake?"]

                INSTRUCTIONS:
                1. Read the question carefully
                2. Identify if there's ONE main topic or MULTIPLE distinct topics
                3. If ONE topic → Simplify keeping the essential
                4. If MULTIPLE topics → Separate and simplify each question
                5. Keep only essential information in each simplified question
                6. reply in english
                QUESTION TO ANALYZE: "{question}"

                MANDATORY response format:
                ANALYSIS: [Single complex question / Multiple distinct questions]
                RESULT: ["simplified question 1", "simplified question 2", ...]

                ANALYSIS:""",

            'ar': f"""أنت خبير في تحليل الأسئلة وتبسيط الأسئلة المعقدة.

                وقت النظام: {current_time}

                المهمة: حلل هذا السؤال وحدد ما إذا كان سؤالاً واحداً معقداً أم عدة أسئلة متميزة.

                القواعد:
                1. إذا كان سؤالاً واحداً معقداً بتفاصيل إضافية حول نفس الموضوع → بسط إلى سؤال قصير واحد
                2. إذا كانت عدة أسئلة متميزة حول مواضيع مختلفة → افصل كل سؤال وبسطها

                أمثلة:

                سؤال معقد واحد:
                "أريد أن أعرف من يشغل حالياً منصب عميد كلية العلوم في جامعة محمد الأول بوجدة. هل يمكنك تقديم اسمه الكامل، خلفيته الأكاديمية والمهنية، تاريخ توليه المنصب، وكذلك وصف لمسؤولياته وإنجازاته ورؤيته للكلية؟"
                → النتيجة: ["من هو عميد كلية العلوم في جامعة محمد الأول بوجدة؟"]

                عدة أسئلة متميزة:
                "من هو عميد كلية العلوم في وجدة؟ وأيضاً، ما هي آخر نتائج فريق برشلونة؟ وكيف أصنع كيكة الشوكولاتة؟"
                → النتيجة: ["من هو عميد كلية العلوم في وجدة؟", "ما هي آخر نتائج برشلونة؟", "كيف أصنع كيكة الشوكولاتة؟"]

                التعليمات:
                1. اقرأ السؤال بعناية
                2. حدد ما إذا كان هناك موضوع رئيسي واحد أم عدة مواضيع متميزة
                3. إذا كان موضوع واحد → بسط مع الاحتفاظ بالأساسي
                4. إذا كانت عدة مواضيع → افصل وبسط كل سؤال
                5. احتفظ فقط بالمعلومات الأساسية في كل سؤال مبسط
                6. الرد باللغة العربية

                السؤال المراد تحليله: "{question}"

                تنسيق الإجابة الإجباري:
                التحليل: [سؤال معقد واحد / عدة أسئلة متميزة]
                النتيجة: ["السؤال المبسط 1", "السؤال المبسط 2", ...]

                التحليل:""",

            'amz': f"""Anta d amussnaw deg usleḍ n isqsiyen i yesseflayen isqsiyen iwuɛren.

                Akud n unagraw: {current_time}

                Tanbaḍt: Sled asqsi-a u ḥded ma yella d asqsi yiwen iwuɛer neɣ deqs n isqsiyen imgerrden.

                Izerfan:
                1. Ma yella d asqsi yiwen iwuɛer s yifutas nniḍen ɣef yiwet n temsalt → Sɣezf ɣer yiwen wasqsi awezlan  
                2. Ma llan deqs n isqsiyen imgerrden ɣef yimḍanen imgerrden → Beṭṭu yal asqsi u sɣezf-iten

                Imedyaten:

                Asqsi iwuɛer yiwen:
                "Bɣiɣ ad ssneɣ anwa i yețțusuddut deg wadda n uεemid n teɣdemt n tussniwin deg tesdawit Mohammed Amezwaru n Wejda. Tzemred ad d-tefked azref-is ummid, abrid-is aɣlnaw d umahal, azemz n tuddut, d ugla n txubbiwin, tiɣawsiwin d tanayrt-is i teɣdemt?"
                → IGMAD: ["Anwa id uεemid n teɣdemt n tussniwin deg Wejda?"]

                Deqs n isqsiyen imgerrden:
                "Anwa id uεemid n teɣdemt n tussniwin n Wejda? Daɣen, d acu id yigmaḍ ineggura n trebbaɛt n tḥarut n Barcelona? D amek ara xdemɣ tikikt n cukula?"
                → IGMAD: ["Anwa id uεemid n teɣdemt n tussniwin n Wejda?", "D acu id yigmaḍ ineggura n Barcelona?", "Amek ara xdemɣ tikikt n cukula?"]

                Tinaḍin:
                1. Ɣer asqsi s tsserti
                2. Sulu ma yella yiwen umḍan agejdan neɣ deqs n yimḍanen imgerrden
                3. Ma yella yiwen umḍan → Sɣezf s uḥraz n lmuhim
                4. Ma llan deqs n yimḍanen → Beṭṭu u sɣezf yal asqsi
                5. Ḥrez kan talɣut tamuhimt deg yal asqsi yețwasɣezfen

                ASQSI I YEȚWASELDEN: "{question}"

                Talɣa n tririt ilaqen:
                ASLEḌ: [Asqsi iwuɛer yiwen / Deqs n isqsiyen imgerrden]
                IGMAD: ["asqsi yețwasɣezfen 1", "asqsi yețwasɣezfen 2", ...]

                ASLEḌ:"""
        }
        
        try:
            prompt = simplification_prompts.get(lang, simplification_prompts['fr'])
            response = self._call_ollama(prompt=prompt)
            
            logger.info(f"Simplification raw response: {response}")
            
            
            simplified_questions = self._extract_simplified_questions(response, lang)
            
            if not simplified_questions:
                simplified_questions = [question.strip()]
            
            classified_questions = []
            for q in simplified_questions:
                classification = self._classify_question_with_temporal_logic(q, lang, date)
                classified_questions.append(classification)
            
            logger.info(f"Classified questions: {classified_questions}")
            return classified_questions
            
        except Exception as e:
            logger.error(f"Erreur lors de la simplification: {str(e)}")
            return [{'question': question.strip(), 'type': 'static', 'reason': 'extraction_error'}]

    def _extract_simplified_questions(self, response: str, lang: str = 'fr') -> list:
        """
        Extrait les questions simplifiées de la réponse du LLM
        """
        import re
        
        simplified_questions = []
        
        try:
            patterns = {
                'fr': [
                    r'RÉSULTAT:\s*\[(.*?)\]',
                    r'résultat:\s*\[(.*?)\]',
                    r'\[(.*?)\]',
                    r'RESULT:\s*\[(.*?)\]',
                    r'result:\s*\[(.*?)\]',
                    r'\[(.*?)\]',
                    r'النتيجة:\s*\[(.*?)\]',
                    r'نتيجة:\s*\[(.*?)\]',
                    r'IGMAD:\s*\[(.*?)\]',
                    r'igmad:\s*\[(.*?)\]',
                ],
                'en': [
                    r'RÉSULTAT:\s*\[(.*?)\]',
                    r'résultat:\s*\[(.*?)\]',
                    r'\[(.*?)\]',
                    r'RESULT:\s*\[(.*?)\]',
                    r'result:\s*\[(.*?)\]',
                    r'\[(.*?)\]',
                    r'النتيجة:\s*\[(.*?)\]',
                    r'نتيجة:\s*\[(.*?)\]',
                    r'IGMAD:\s*\[(.*?)\]',
                    r'igmad:\s*\[(.*?)\]',
                ],
                'ar': [
                    r'RÉSULTAT:\s*\[(.*?)\]',
                    r'résultat:\s*\[(.*?)\]',
                    r'\[(.*?)\]',
                    r'RESULT:\s*\[(.*?)\]',
                    r'result:\s*\[(.*?)\]',
                    r'\[(.*?)\]',
                    r'النتيجة:\s*\[(.*?)\]',
                    r'نتيجة:\s*\[(.*?)\]',
                    r'IGMAD:\s*\[(.*?)\]',
                    r'igmad:\s*\[(.*?)\]',
                ],
                'amz': [
                    r'RÉSULTAT:\s*\[(.*?)\]',
                    r'résultat:\s*\[(.*?)\]',
                    r'\[(.*?)\]',
                    r'RESULT:\s*\[(.*?)\]',
                    r'result:\s*\[(.*?)\]',
                    r'\[(.*?)\]',
                    r'النتيجة:\s*\[(.*?)\]',
                    r'نتيجة:\s*\[(.*?)\]',
                    r'IGMAD:\s*\[(.*?)\]',
                    r'igmad:\s*\[(.*?)\]',
                ]
            }
            
            current_patterns = patterns.get(lang, patterns['fr'])
            
            for pattern in current_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    questions_text = match.group(1)
                    break
            else:
                questions_text = response
            
            
            question_matches = re.findall(r'"([^"]+)"', questions_text)
            
            if question_matches:
                simplified_questions = [q.strip() for q in question_matches if q.strip()]
            else:
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if any(marker in line.lower() for marker in ['•', '-', '1.', '2.', '3.']) or line.endswith('?'):
                        # Nettoyer la ligne
                        clean_line = re.sub(r'^[\s\-•\d\.]+', '', line).strip()
                        if clean_line and len(clean_line) > 5:
                            simplified_questions.append(clean_line)
            
            return simplified_questions[:5]  # Limiter à 5 questions max
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {str(e)}")
            return []

    def _classify_question_with_temporal_logic(self, question: str, lang: str, reference_date: datetime) -> dict:
        """
        Classifie une question comme statique ou dynamique avec logique temporelle avancée
        """
        import re
        
        current_time = datetime.now()
        
        year_pattern = r'\b(20\d{2})[^\d]'
        semester_pattern = r'\b(semestre?|semester)\s*(\d+)?\s*(20\d{2})[-/]?(20\d{2})?\b'
        academic_year_pattern = r'\b(20\d{2})[-/](20\d{2})\b'
        
        question_lower = question.lower()
        
        # 1. RÈGLES SPÉCIALES POUR FACULTÉ DES SCIENCES OUJDA
        faculty_sciences_keywords = [
            'faculté des sciences', 'faculty of sciences', 'كلية العلوم', 'teɣdemt n tussniwin',
            'math', 'mathématiques', 'mathematics', 'الرياضيات',
            'physics', 'physique', 'الفيزياء', 'tafizikt',
            'chemistry', 'chimie', 'الكيمياء', 'tikimi',
            'informatique', 'computer science', 'علوم الحاسوب', 'tasenselkimt',
            'biology', 'biologie', 'الأحياء', 'tasnudert',
            'geology', 'géologie', 'الجيولوجيا', 'tarakalt'
        ]
        
        is_faculty_sciences = any(keyword in question_lower for keyword in faculty_sciences_keywords)
        
        # 2. VÉRIFICATION DES POSTES (DOYENS, DIRECTEURS, ETC.)
        position_keywords = ['doyen', 'dean', 'عميد', 'directeur', 'director', 'مدير', 'responsable', 'chef']
        is_position_question = any(keyword in question_lower for keyword in position_keywords)
        
        if is_position_question:
            # Pour les questions de postes, vérifier l'âge hypothétique du mandat
            # Si c'est la faculté des sciences, supposer que si > 4 ans → dynamique
            time_since_reference = current_time - reference_date
            if time_since_reference.days > (4 * 365):  # Plus de 4 ans
                return {
                    'question': question,
                    'type': 'dynamic',
                    'reason': f'Position question older than 4 years (reference: {reference_date.strftime("%Y-%m-%d")})'
                }
            else:
                return {
                    'question': question,
                    'type': 'static',
                    'reason': f'Position question within 4 years (reference: {reference_date.strftime("%Y-%m-%d")})'
                }
        
        # 3. VÉRIFICATION DES EMPLOIS DU TEMPS / HORAIRES
        schedule_keywords = ['emploi du temps', 'schedule', 'timetable', 'جدول', 'horaire', 'planning']
        is_schedule_question = any(keyword in question_lower for keyword in schedule_keywords)
        
        if is_schedule_question:
            # Chercher des années spécifiques dans la question
            year_matches = re.findall(year_pattern, question)
            semester_matches = re.findall(semester_pattern, question, re.IGNORECASE)
            academic_year_matches = re.findall(academic_year_pattern, question)
            
            if year_matches or semester_matches or academic_year_matches:
                # Extraire l'année la plus récente mentionnée
                years = []
                for match in year_matches:
                    years.append(int(match))
                
                for match in semester_matches:
                    if len(match) >= 3 and match[2]:  # Année dans le match
                        years.append(int(match[2]))
                
                for match in academic_year_matches:
                    years.extend([int(match[0]), int(match[1])])
                
                if years:
                    latest_year = max(years)
                    mentioned_date = datetime(latest_year, 9, 1)  # Supposer septembre comme début d'année académique
                    
                    # Si plus de 5 mois (environ un semestre)
                    if (current_time - mentioned_date).days > 150:
                        return {
                            'question': self._update_question_with_current_time(question, current_time),
                            'type': 'dynamic',
                            'reason': f'Schedule from {latest_year} is more than 5 months old'
                        }
                    else:
                        return {
                            'question': question,
                            'type': 'static',
                            'reason': f'Schedule from {latest_year} is still current'
                        }
            else:
                return {
                    'question': question,
                    'type': 'dynamic',
                    'reason': 'Current schedule question without specific year'
                }
        
        dynamic_indicators = [
            'actuellement', 'currently', 'حالياً', 'tura',
            'récent', 'recent', 'حديث', 'amaynu',
            'nouveau', 'new', 'جديد', 'amaynu',
            'dernière', 'latest', 'آخر', 'aneggaru',
            'maintenant', 'now', 'الآن', 'tura',
            "aujourd'hui", 'today', 'اليوم', 'ass-a',
            'cette année', 'this year', 'هذه السنة',
            'disponible', 'available', 'متوفر', 'yella'
        ]
        
        static_indicators = [
            'comment', 'how', 'كيف', 'amek',
            "qu'est-ce que", 'what is', 'ما هو', 'd acu id',
            'définition', 'definition', 'تعريف', 'asbadu',
            'histoire', 'history', 'تاريخ', 'amezruy',
            'procédure', 'procedure', 'إجراء', 'tarrayt'
        ]
        
        # Compter les indicateurs
        dynamic_count = sum(1 for indicator in dynamic_indicators if indicator in question_lower)
        static_count = sum(1 for indicator in static_indicators if indicator in question_lower)
        
        # 5. DÉCISION FINALE
        if dynamic_count > static_count:
            return {
                'question': question,
                'type': 'dynamic',
                'reason': f'Dynamic indicators detected: {dynamic_count} vs static: {static_count}'
            }
        elif static_count > 0:
            return {
                'question': question,
                'type': 'static',
                'reason': f'Static indicators detected: {static_count} vs dynamic: {dynamic_count}'
            }
        else:
            # Par défaut, considérer comme statique pour les questions générales
            return {
                'question': question,  
                'type': 'static',
                'reason': 'No clear temporal indicators, defaulting to static'
            }

    def _update_question_with_current_time(self, question: str, current_time: datetime) -> str:
        """
        Met à jour une question avec l'année/période actuelle
        """
        import re
        
        current_year = current_time.year
        current_academic_year = f"{current_year}-{current_year + 1}" if current_time.month >= 9 else f"{current_year - 1}-{current_year}"
        
        # Remplacer les années spécifiques par l'année académique actuelle
        question = re.sub(r'\b20\d{2}[-/]20\d{2}\b', current_academic_year, question)
        question = re.sub(r'\b(semestre?|semester)\s*\d+\s*20\d{2}[-/]?20\d{2}?\b', 
                        f'semestre actuel {current_academic_year}', question, flags=re.IGNORECASE)
        
        return question

    def _format_comprehensive_qa_pairs(self, question_answer_pairs: List[Dict], lang: str) -> str:
        """Format question-answer pairs for comprehensive processing"""
        formatted_pairs = []
        
        for i, pair in enumerate(question_answer_pairs, 1):
            question = pair['question']
            documents = pair['documents']
            
            # Format documents for this question
            if documents:
                answers = []
                for doc in documents:
                    answer_text = doc.get('answer', 'N/A')
                    confidence = doc.get('confidence', 0.0)
                    date = doc.get('date', 'N/A')
                    answers.append(f"  • {answer_text} (Confiance: {confidence:.2f}, Date: {date})")
                
                formatted_pair = f"""Question {i}: {question}
                                Réponses disponibles:
                                {chr(10).join(answers)}"""
            else:
                formatted_pair = f"""Question {i}: {question}
                                Réponses disponibles: Aucune réponse trouvée"""
            
            formatted_pairs.append(formatted_pair)
        
        return "\n\n".join(formatted_pairs)
    
    def _calculate_comprehensive_confidence(self, question_answer_pairs: List[Dict], all_documents: List[Dict]) -> float:
        """Calculate confidence for comprehensive response"""
        if not question_answer_pairs:
            return 0.0
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for pair in question_answer_pairs:
            documents = pair['documents']
            if documents:
                # Calculate average confidence for this question's documents
                doc_confidences = [doc.get('confidence', 0.0) for doc in documents]
                avg_confidence = sum(doc_confidences) / len(doc_confidences)
                
                # Weight by number of documents (more documents = higher weight)
                weight = min(len(documents), 3)  # Cap at 3 for diminishing returns
                
                total_confidence += avg_confidence * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = total_confidence / total_weight
        
        # Boost confidence if we have good coverage of questions
        coverage_bonus = len([p for p in question_answer_pairs if p['documents']]) / len(question_answer_pairs)
        
        final_confidence = min(base_confidence * (0.7 + 0.3 * coverage_bonus), 1.0)
        
        return final_confidence

    def _analyze_question_coverage(self, question_answer_pairs: List[Dict], response: str) -> Dict[str, Any]:
        """Analyze how well the response covers each question"""
        coverage_analysis = {
            'total_questions': len(question_answer_pairs),
            'questions_with_data': len([p for p in question_answer_pairs if p['documents']]),
            'coverage_percentage': 0.0,
            'question_details': []
        }
        
        for pair in question_answer_pairs:
            question_coverage = {
                'question': pair['question'],
                'has_documents': bool(pair['documents']),
                'num_documents': len(pair['documents']),
                'intent': pair['intent'],
                'appears_answered': len(pair['question'].split()) > 2 and any(
                    word.lower() in response.lower() 
                    for word in pair['question'].split()[:3]
                )
            }
            coverage_analysis['question_details'].append(question_coverage)
        
        if coverage_analysis['total_questions'] > 0:
            coverage_analysis['coverage_percentage'] = (
                coverage_analysis['questions_with_data'] / coverage_analysis['total_questions']
            ) * 100
        
        return coverage_analysis

    def _parse_validation_response(self, validation_response: str) -> Dict[str, Any]:
        """Parse LLM validation response into structured format"""
        try:
            # Look for key indicators in the response
            response_lower = validation_response.lower()
            
            # Simple heuristic validation
            is_valid = "valid: 1" in response_lower or "satisfactory" in response_lower
            
            return {
                "is_valid": is_valid,
                "coverage_score": 0.8 if is_valid else 0.3,
                "missing_aspects": [],
                "irrelevant_content": [],
                "raw_response": validation_response
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "coverage_score": 0.0,
                "missing_aspects": ["parsing_error"],
                "irrelevant_content": [],
                "error": str(e)
            }

    def generate_comprehensive_response_optimized(self, original_question: str, question_answer_pairs: List[Dict], all_documents: List[Dict], lang: str, validate_and_fallback: bool = True) -> Dict[str, Any]:
        """
        OPTIMIZED: Single LLM call that generates response AND validates AND handles fallback
        """
        try:
            start_time = datetime.now()
            
            optimized_prompts = {
                'fr': {
                    'system': """Tu es un expert en synthèse d'informations pour la Faculté des Sciences d'Oujda (FSO). 
                    Ta tâche est d'analyser plusieurs paires question-réponse et de générer une réponse comprehensive.

                    INSTRUCTIONS IMPORTANTES:
                    1. Analyse TOUTES les questions et leurs réponses
                    2. Si les réponses ne sont PAS pertinentes pour les questions, indique "IRRELEVANT_CONTENT" au début
                    3. Génère une réponse cohérente qui traite tous les aspects
                    4. Combine les informations de différentes sources (base de données + internet)
                    5. Résous les conflits entre réponses
                    6. Indique clairement les sources d'information
                    7. Pour les informations temporelles, précise la période
                    8. reponds en français
                    FORMAT DE RÉPONSE:
                    - Si contenu non pertinent: commence par "IRRELEVANT_CONTENT"
                    - Sinon: génère directement la réponse comprehensive

                    CONTEXTE FSO: Faculté des Sciences Oujda, Université Mohammed Premier""",

                    'user': """QUESTION ORIGINALE: {original_question}

                    QUESTIONS ET RÉPONSES DISPONIBLES:
                    {formatted_qa_pairs}

                    CONTEXTE: {num_questions} questions, {num_sources} sources (base de données + internet)

                    GÉNÈRE une réponse comprehensive qui traite tous les aspects. Si les réponses ne sont pas pertinentes aux questions, commence par "IRRELEVANT_CONTENT"."""
                },
                
                'en': {
                    'system': """You are an expert information synthesizer for the Faculty of Sciences Oujda (FSO). 
                    Your task is to analyze multiple question-answer pairs and generate a comprehensive response.

                    IMPORTANT INSTRUCTIONS:
                    1. Analyze ALL questions and their answers
                    2. If answers are NOT relevant to questions, indicate "IRRELEVANT_CONTENT" at the beginning
                    3. Generate a coherent response addressing all aspects
                    4. Combine information from different sources (database + internet)
                    5. Resolve conflicts between answers
                    6. Clearly indicate information sources
                    7. For temporal information, specify the time period
                    8. response in english
                    RESPONSE FORMAT:
                    - If irrelevant content: start with "IRRELEVANT_CONTENT"
                    - Otherwise: generate comprehensive response directly

                    FSO CONTEXT: Faculty of Sciences Oujda, Mohammed First University""",

                    'user': """ORIGINAL QUESTION: {original_question}

                    AVAILABLE QUESTIONS AND ANSWERS:
                    {formatted_qa_pairs}

                    CONTEXT: {num_questions} questions, {num_sources} sources (database + internet)

                    GENERATE a comprehensive response addressing all aspects. If answers are not relevant to questions, start with "IRRELEVANT_CONTENT"."""
                },
                'ar': {
                    'system': """أنت مُركّب خبير للمعلومات بكلية العلوم وجدة (FSO). 
                    مهمتك هي تحليل أزواج متعددة من الأسئلة والأجوبة وإنشاء رد شامل.

                    تعليمات مهمة:
                    1. حلل جميع الأسئلة والأجوبة الخاصة بها
                    2. إذا كانت الأجوبة غير مرتبطة بالأسئلة، اذكر "IRRELEVANT_CONTENT" في البداية
                    3. أنشئ ردًا مترابطًا يتناول جميع الجوانب
                    4. اجمع المعلومات من مصادر مختلفة (قاعدة البيانات + الإنترنت)
                    5. حل التعارضات بين الأجوبة
                    6. حدد مصادر المعلومات بوضوح
                    7. بالنسبة للمعلومات الزمنية، حدد الفترة الزمنية
                    8. الرد باللغة العربية

                    تنسيق الرد:
                    - إذا كان المحتوى غير مرتبط: ابدأ بـ "IRRELEVANT_CONTENT"
                    - وإلا: أنشئ ردًا شاملًا مباشرةً

                    سياق كلية العلوم وجدة: كلية العلوم وجدة، جامعة محمد الأول""",

                    'user': """السؤال الأصلي: {original_question}

                    الأسئلة والأجوبة المتاحة:
                    {formatted_qa_pairs}

                    السياق: {num_questions} أسئلة، {num_sources} مصادر (قاعدة البيانات + الإنترنت)

                    أنشئ ردًا شاملًا يتناول جميع الجوانب. إذا كانت الأجوبة غير مرتبطة بالأسئلة، ابدأ بـ "IRRELEVANT_CONTENT"."""
                }
            }
            
            prompt_config = optimized_prompts.get(lang, optimized_prompts['fr'])
            
            # Format question-answer pairs efficiently
            formatted_pairs = self._format_comprehensive_qa_pairs_optimized(question_answer_pairs, lang)
            
            # Create the comprehensive prompt
            user_prompt = prompt_config['user'].format(
                original_question=original_question,
                formatted_qa_pairs=formatted_pairs,
                num_questions=len(question_answer_pairs),
                num_sources=len(all_documents)
            )
            
            # SINGLE LLM CALL that does everything
            comprehensive_response = self._call_ollama(
                prompt=user_prompt,
                system_prompt=prompt_config['system']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Check if LLM detected irrelevant content
            used_fallback = False
            if comprehensive_response.startswith("IRRELEVANT_CONTENT"):
                if validate_and_fallback:
                    logger.info("LLM detected irrelevant content, performing internet fallback")
                    # Perform internet search for questions that had poor database results
                    fallback_response = self._perform_internet_fallback(question_answer_pairs, lang)
                    if fallback_response:
                        comprehensive_response = fallback_response
                        used_fallback = True
                    else:
                        comprehensive_response = comprehensive_response.replace("IRRELEVANT_CONTENT", "").strip()
                else:
                    comprehensive_response = comprehensive_response.replace("IRRELEVANT_CONTENT", "").strip()
            
            # Calculate confidence efficiently
            confidence = self._calculate_confidence_fast(question_answer_pairs, all_documents)
            
            logger.info(f"Optimized comprehensive response generated with confidence: {confidence}")
            
            return {
                'response': comprehensive_response,
                'confidence': confidence,
                'sources_used': len(all_documents),
                'questions_addressed': len(question_answer_pairs),
                'processing_time': processing_time,
                'used_fallback': used_fallback,
                'scope': 'optimized_comprehensive'
            }
            
        except Exception as e:
            logger.error(f"Error generating optimized comprehensive response: {str(e)}")
            return {
                'response': self.no_results_messages.get(lang, 'Erreur lors du traitement.'),
                'confidence': 0.0,
                'sources_used': 0,
                'questions_addressed': 0,
                'processing_time': 0,
                'used_fallback': False,
                'error': str(e),
                'scope': 'error'
            }
    
    def _perform_internet_fallback(self, question_answer_pairs: List[Dict], lang: str) -> str:
        """
        Perform internet search fallback for questions with poor database results
        """
        try:
            fallback_questions = []
            
            # Identify questions that need internet fallback
            for pair in question_answer_pairs:
                if pair['source'] == 'database' and len(pair['documents']) < 2:
                    # Database results are sparse, try internet
                    fallback_questions.append(pair['question'])
            
            if not fallback_questions:
                return None
            
            logger.info(f"Performing internet fallback for {len(fallback_questions)} questions")
            
            # Get internet results for these questions
            all_internet_results = []
            for question in fallback_questions:
                internet_results = get_internet_results_for_question(question, lang)
                all_internet_results.extend(internet_results)
            
            if not all_internet_results:
                return None
            
            # Use existing internet function to generate response
            internet_response = get_internet_results_for_question(fallback_questions, lang)
            return internet_response.get('structured_response', '')
            
        except Exception as e:
            logger.error(f"Error in internet fallback: {str(e)}")
            return None

    def _format_comprehensive_qa_pairs_optimized(self, question_answer_pairs: List[Dict], lang: str) -> str:
        """Optimized formatting that's more concise"""
        formatted_pairs = []
        
        for i, pair in enumerate(question_answer_pairs, 1):
            question = pair['question']
            documents = pair['documents']
            source = pair['source']
            
            if documents:
                top_docs = documents[:2]
                answers_text = " | ".join([doc.get('answer', '')[:200] + "..." if len(doc.get('answer', '')) > 200 else doc.get('answer', '') for doc in top_docs])
                formatted_pair = f"Q{i} ({source}): {question}\nA{i}: {answers_text}"
            else:
                formatted_pair = f"Q{i}: {question}\nA{i}: Aucune réponse trouvée"
            
            formatted_pairs.append(formatted_pair)
        
        return "\n\n".join(formatted_pairs)

    def _calculate_confidence_fast(self, question_answer_pairs: List[Dict], all_documents: List[Dict]) -> float:
        """Fast confidence calculation without complex logic"""
        if not question_answer_pairs:
            return 0.0
        
        questions_with_docs = len([p for p in question_answer_pairs if p['documents']])
        coverage_ratio = questions_with_docs / len(question_answer_pairs)
        
        doc_confidences = [doc.get('confidence', 0.5) for doc in all_documents if 'confidence' in doc]
        avg_doc_confidence = sum(doc_confidences) / len(doc_confidences) if doc_confidences else 0.5
        
        final_confidence = (coverage_ratio * 0.6) + (avg_doc_confidence * 0.4)
        
        return min(final_confidence, 1.0)


llm_service = LLMService()
