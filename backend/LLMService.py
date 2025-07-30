# -------------------------------  GPU ------------------------------- #

import requests
import logging
from typing import List, Dict, Any, Union
from pydantic import BaseModel
import json
import os
import uuid
from pathlib import Path
from datetime import datetime
import psutil
import GPUtil

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        
        # Configuration GPU optimisée pour RTX 3050 (4GB VRAM)
        # IMPORTANT: Forcer l'utilisation de la RTX 3050 (GPU 1)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3050 est le GPU 1
        os.environ['OLLAMA_GPU_LAYERS'] = '999'  # Toutes les couches sur GPU
        os.environ['OLLAMA_NUM_PARALLEL'] = '1'  # Une seule inférence à la fois
        os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Un seul modèle en mémoire
        os.environ['OLLAMA_KEEP_ALIVE'] = '10m'  # Garde le modèle en mémoire plus longtemps
        
        # Configuration spécifique NVIDIA
        os.environ['NVIDIA_VISIBLE_DEVICES'] = '1'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # Configuration pour Ollama
        self.base_url = "http://localhost:11434"
        self.model_name = "llama3:8b"
        
        # Paramètres optimisés pour GPU
        self.gpu_optimized_options = {
            "num_ctx": 2048,  # Contexte réduit pour économiser VRAM
            "num_batch": 512,  # Batch size optimisé
            "num_gqa": 8,      # Grouped Query Attention
            "num_gpu": 999,    # Toutes les couches sur GPU
            "num_thread": 4,   # Threads CPU pour les ops non-GPU
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "num_predict": 1200,
            "use_mmap": True,   # Memory mapping pour efficacité
            "use_mlock": True,  # Lock memory pour performance
        }

        # Prompts optimisés pour la structuration de réponses
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
            # Payload optimisé pour GPU
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": self.gpu_optimized_options.copy()
            }
            
            # Ajouter le system prompt si fourni
            if system_prompt:
                payload["system"] = system_prompt
            
            # Log avant l'appel
            start_time = datetime.now()
            logger.info(f"Appel Ollama GPU - Prompt: {len(prompt)} caractères")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60000  # Timeout réduit car GPU est plus rapide
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Log des performances
                logger.info(f"Réponse générée en {processing_time:.2f}s")
                logger.info(f"Tokens évalués: {result.get('eval_count', 'N/A')}")
                logger.info(f"Vitesse: {result.get('eval_count', 0) / processing_time:.1f} tokens/s")
                
                return response_text
            else:
                raise Exception(f"Erreur Ollama HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de l'appel à Ollama: {str(e)}")

    def validate_answer_relevance(self, question: str, answer: str) -> bool:
        """Validate if the answer is relevant to the question using LLM"""
        
        system_prompt = """You are an intelligent answer validator. Your task is to determine if a given answer is relevant and correct for a specific question.

    Rules:
    1. Return only "1" if the answer is relevant and addresses the question
    2. Return only "0" if the answer is irrelevant, incorrect, or doesn't address the question
    3. Be contextually aware: FSO = Faculté des Sciences Oujda
    4. Consider partial matches: if question asks "le doyen" and answer mentions "Doyen de la Faculté des Sciences d'Oujda", this is relevant
    5. No explanations, just return 1 or 0"""

        user_prompt = f"""Question: {question}
    Answer: {answer}

    Is this answer relevant and correct for the question? Return only 1 or 0."""

        try:
            llm_response = self._call_ollama(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            # Debug logging
            logger.info(f"Validation - Question: {question}")
            logger.info(f"Validation - Answer preview: {answer[:100]}...")
            logger.info(f"Validation - LLM raw response: '{llm_response}'")
            
            # Extract only the first character and validate
            result = llm_response.strip()[0] if llm_response.strip() else "0"
            logger.info(f"Validation - Extracted result: '{result}'")
            
            # Return boolean based on LLM response
            is_valid = result == "1"
            logger.info(f"Validation - Final result: {is_valid}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating answer relevance: {str(e)}")
            # Default to False if validation fails
            return False
        
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

    def generate_structured_response(self, question: str, search_results: List[Dict[str, Any]], lang: str) -> Dict[str, Any]:
        """Génère une réponse structurée à partir de TOUS les résultats trouvés"""
        
        try:
            start_time = datetime.now()
            
            # Filter valid results
            valid_results = [r for r in search_results if r.get('answer')]
            
            if not valid_results:
                return {
                    'response': self.no_results_messages.get(lang, self.no_results_messages.get('fr', 'Aucune réponse trouvée.')),
                    'confidence': 0.0,
                    'sources_used': 0,
                    'processing_time': 0,
                    'scope': 'no_results'
                }
            
            # Use prompts according to language
            prompt_config = self.prompts.get(lang, self.prompts.get('fr', self.prompts['fr']))
            
            # Format ALL results for the LLM
            formatted_results = self.format_search_results_for_structuring(valid_results)
            
            # Create complete prompt
            user_prompt = prompt_config['user'].format(
                question=question,
                search_results=formatted_results
            )
            
            # Call Ollama to structure the response
            structured_response = self._call_ollama(
                prompt=user_prompt,
                system_prompt=prompt_config['system']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence based on number and quality of results
            confidence = self._calculate_confidence(valid_results)
            
            logger.info(f"Response generated with confidence: {confidence}")
            
            return {
                'response': structured_response,
                'confidence': confidence,
                'sources_used': len(valid_results),
                'processing_time': processing_time,
                'original_results': valid_results,
                'scope': 'fso_related'
            }
            
        except Exception as e:
            logger.error(f"Error generating structured response: {str(e)}")
            return {
                'response': self.no_results_messages.get(lang, self.no_results_messages.get('fr', 'Erreur lors du traitement.')),
                'confidence': 0.0,
                'sources_used': 0,
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
            
            # Bonus pour plusieurs sources
            source_bonus = min(len(results) * 0.05, 0.2)
            
            # Bonus si les scores sont élevés
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
            
            # Ajouter info GPU si disponible
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
                    if len(value) <= 5:  # Limit list items to avoid overwhelming the model
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

            Question: {question}

            Precise and factual response based on your FSO training data:""",

                    'ar': f"""أنت المساعد الافتراضي الرسمي لكلية العلوم بوجدة (FSO) بجامعة محمد الأول.

            تم تدريبك خصيصاً على بيانات كلية العلوم. استخدم فقط معرفتك بكلية العلوم للإجابة.

            كلية العلوم فقط:
            - كلية العلوم بوجدة، جامعة محمد الأول، المغرب
            - تجنب الخلط مع كليات أخرى (الآداب، الاقتصاد، إلخ)
            - أجب فقط عما يخص كلية العلوم

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
                'confidence': 0.9,  # Plus de confiance avec le modèle fine-tuné
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
        """Process SERP avec validation FSO et integration du modèle fine-tuné"""
        
        try:
            # Validation FSO dans les données SERP
            fso_indicators = [
                "faculté des sciences", "fso", "sciences oujda", 
                "mohammed premier", "ump.ac.ma", "fso.ump.ma"
            ]
            
            serp_text = str(serp_data).lower()
            fso_score = sum(1 for indicator in fso_indicators if indicator in serp_text)
            
            # Si peu d'indicateurs FSO, utiliser le modèle fine-tuné directement
            if fso_score < 2:
                logger.warning("SERP data contains little FSO content, using fine-tuned model instead")
                return self.generate_faculty_response(question, lang)
            
            # System prompt amélioré pour traitement SERP
            system_prompt = (
                "TASK: Process SERP data for Faculty of Sciences Oujda (FSO) ONLY\n"
                "CRITICAL: Filter out any content from other faculties (Letters, Economics, EST, etc.)\n"
                "1. Use EXACT client question - DO NOT rephrase\n"
                "2. Extract ONLY FSO-relevant facts from SERP data\n"
                "3. Reject any information about other faculties\n"
                "4. If no FSO-specific info, indicate clearly\n"
                "Language: " + str(lang) + "\n"
            )
            
            # Template JSON avec validation FSO
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
            
            # Format SERP data avec pre-filtering
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
            
            # Call Ollama avec votre modèle fine-tuné
            payload = {
                "model": self.model_name,  # Votre modèle fine-tuné
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
                
                # Parse et validate JSON
                json_start = llm_output.find('{')
                json_end = llm_output.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = llm_output[json_start:json_end]
                    processed = json.loads(json_str)
                    
                    # Validation FSO
                    validation = processed.get('validation', {})
                    if not validation.get('is_fso_relevant', False):
                        logger.warning("SERP content not FSO-relevant, falling back to fine-tuned model")
                        return self.generate_faculty_response(question, lang)
                    
                    knowledge_entry = processed.get('knowledge_entry', {})
                    
                    # Force original question preservation
                    questions = knowledge_entry.get('question', {})
                    if questions.get('fr') and isinstance(questions['fr'], list):
                        if questions['fr'][0] != question:
                            questions['fr'][0] = question
                    
                    # Store to file if requested
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
            # Fallback to fine-tuned model
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
            
            # Post-traitement pour s'assurer qu'il n'y a pas de doublons
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
            # D'abord, supprimer les blocs complets dupliqués
            text = self._remove_block_duplicates(text)
            
            # Ensuite, supprimer les phrases dupliquées
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
            # Diviser par paragraphes ou sections
            paragraphs = text.split('\n\n')
            unique_paragraphs = []
            seen_paragraphs = set()
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph:
                    # Normaliser pour comparaison (sans espaces multiples, minuscules)
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
        Filtre le contexte pour se concentrer sur FSO plutôt que CAP-FSO
        """
        try:
            filtered_context = {}
            
            # Mots-clés prioritaires pour FSO
            fso_keywords = [
                'faculté des sciences',
                'faculty of sciences',
                'fso',
                'oujda',
                'université mohammed premier',
                'mohammed first university',
                'ump'
            ]
            
            # Mots-clés à éviter ou minimiser
            avoid_keywords = [
                'cap-fso',
                'cap fso',
                'commission académique'
            ]
            
            for key, value in context.items():
                if isinstance(value, str):
                    # Priorité aux contenus mentionnant FSO
                    if any(keyword in value.lower() for keyword in fso_keywords):
                        # Éviter les contenus trop centrés sur CAP-FSO
                        if not any(avoid in value.lower() for avoid in avoid_keywords):
                            filtered_context[key] = value
                        elif any(fso_kw in value.lower() for fso_kw in fso_keywords[:3]):
                            # Garde le contenu s'il mentionne aussi FSO directement
                            filtered_context[key] = value
                else:
                    filtered_context[key] = value
            
            return filtered_context if filtered_context else context
            
        except Exception as e:
            logger.error(f"Erreur lors du filtrage du contexte: {str(e)}")
            return context

    def build_enhanced_serp_query(self, question: str, lang: str = 'fr') -> str:
        """Construit une requête SERP améliorée pour éviter les autres facultés"""
        
        # Sites spécifiques FSO + exclusion explicite des autres facultés
        fso_sites = [
            "site:fso.ump.ma",
            "site:ump.ac.ma/fso", 
            "site:sciences.ump.ac.ma"
        ]
        
        # Exclusions explicites pour éviter autres facultés
        exclude_sites = [
            "-site:flsh.ump.ac.ma",     # Faculté Lettres
            "-site:est.ump.ac.ma",      # EST
            "-site:encg.ump.ac.ma",     # ENCG
            "-site:fsjes.ump.ac.ma",    # Économie/Droit
            "-inurl:lettres",
            "-inurl:economie", 
            "-inurl:droit",
            "-inurl:est"
        ]
        
        # Mots-clés de renforcement FSO
        fso_keywords = {
            'fr': ['"faculté sciences"', '"FSO"', '"sciences oujda"'],
            'en': ['"faculty sciences"', '"FSO"', '"sciences oujda"'],
            'ar': ['"كلية العلوم"', '"العلوم وجدة"'],
            'amz': ['"tasnawalt tussniwin"']
        }
        
        # Construction de la requête
        sites_part = " OR ".join(fso_sites)
        exclude_part = " ".join(exclude_sites)
        keywords = " ".join(fso_keywords.get(lang, fso_keywords['fr']))
        
        # Requête finale optimisée
        enhanced_query = f"({sites_part}) {keywords} {question} {exclude_part}"
        
        logger.info(f"Enhanced SERP query: {enhanced_query}")
        return enhanced_query

    def _filter_fso_content(self, serp_data: str) -> str:
        """Filtre le contenu SERP pour garder seulement les données FSO"""
        
        if isinstance(serp_data, dict):
            serp_data = str(serp_data)
        
        # Indicateurs positifs FSO
        fso_positive = [
            "faculté des sciences", "fso", "sciences oujda", 
            "ump.ac.ma", "fso.ump.ma", "mohammed premier"
        ]
        
        # Indicateurs négatifs (autres facultés)
        fso_negative = [
            "faculté des lettres", "flsh", "économie", "fsjes",
            "est oujda", "encg", "droit", "lettres"
        ]
        
        lines = serp_data.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Vérifier indicateurs négatifs
            has_negative = any(neg in line_lower for neg in fso_negative)
            if has_negative:
                continue
                
            # Vérifier indicateurs positifs ou ligne neutre
            has_positive = any(pos in line_lower for pos in fso_positive)
            if has_positive or len(line.strip()) < 50:  # Lignes courtes probablement neutres
                filtered_lines.append(line)
        
        filtered_content = '\n'.join(filtered_lines)
        
        # Limite la taille pour éviter les timeouts
        if len(filtered_content) > 2000:
            filtered_content = filtered_content[:2000] + "... [filtered and truncated]"
        
        logger.info(f"Filtered SERP content: {len(serp_data)} -> {len(filtered_content)} chars")
        return filtered_content

    def get_hybrid_response(self, question: str, lang: str = 'fr') -> Dict[str, Any]:
        """Méthode hybride: modèle fine-tuné d'abord, SERP en fallback"""
        
        # 1. Essayer d'abord le modèle fine-tuné
        logger.info("Trying fine-tuned model first...")
        finetuned_response = self.generate_faculty_response(question, lang)
        
        # 2. Vérifier la qualité de la réponse
        response_text = finetuned_response.get('response', '').lower()
        
        # Indicateurs de réponse faible
        weak_indicators = [
            "je ne sais pas", "don't know", "لا أعرف", "ur ẓriɣ ara",
            "pas d'information", "no information", "لا توجد معلومات",
            "désolé", "sorry", "آسف", "suref"
        ]
        
        has_weak_response = any(indicator in response_text for indicator in weak_indicators)
        is_too_short = len(response_text.strip()) < 50
        
        # 3. Si réponse faible, utiliser SERP en complément
        if has_weak_response or is_too_short:
            logger.info("Fine-tuned response weak, trying SERP enhancement...")
            
            try:
                # Construire requête SERP améliorée
                enhanced_query = self.build_enhanced_serp_query(question, lang)
                serp_data = self.search_web(enhanced_query)  # Votre méthode de recherche
                
                if serp_data:
                    serp_response = self.process_serp_to_response(question, serp_data, lang)
                    
                    # Combiner les deux réponses si SERP apporte du contenu
                    if serp_response.get('confidence', 0) > 0.5:
                        return {
                            **serp_response,
                            'source': 'hybrid_finetuned_serp',
                            'fallback_used': True
                        }
            
            except Exception as e:
                logger.warning(f"SERP fallback failed: {str(e)}")
        
        # 4. Retourner la réponse du modèle fine-tuné
        return {
            **finetuned_response,
            'fallback_used': False
        }

    def classify_question_type(self, question: str, lang: str = 'fr') -> str:
        """
        Classifie si une question (quelque soit le domaine) nécessite une réponse STATIQUE ou DYNAMIQUE
        Returns: 'static' or 'dynamic'
        """
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        
        classification_prompts = {
            'fr': f"""Tu es un expert qui classifie tous types de questions selon leur nature temporelle.

    TEMPS SYSTÈME: {current_time}
    BASE DE CONNAISSANCES: Juillet 1, 2025

    TÂCHE: Détermine si cette question nécessite une réponse STATIQUE ou DYNAMIQUE.

    DÉFINITIONS:

    STATIQUE = Informations fixes, stables, factuelles qui ne changent pas souvent:
    - Définitions, concepts, théories
    - Faits historiques et géographiques
    - Procédures générales et règlements établis
    - Structures organisationnelles de base
    - Informations biographiques établies
    - Connaissances scientifiques établies
    - Méthodes et techniques générales
    - Informations institutionnelles fondamentales
    - Tout ce qui reste vrai indépendamment du temps

    DYNAMIQUE = Informations qui changent avec le temps ou nécessitent des données actuelles/récentes:
    - Événements actuels, nouvelles, actualités
    - DATES SPÉCIFIQUES, horaires, calendriers, délais, échéances
    - Statistiques et données actuelles
    - Nominations, changements récents
    - Disponibilité actuelle de services/produits
    - Prix, cotations, marchés actuels
    - Météo, conditions actuelles
    - Informations "en temps réel"
    - Publications, sorties récentes
    - DATES D'INSCRIPTION, deadlines, périodes d'ouverture
    - Tout ce qui nécessite des infos après juillet 2025
    - Questions avec mots-clés temporels: "récent", "nouveau", "actuel", "dernier", "maintenant", "aujourd'hui", "date de", "quand", "deadline"

    EXEMPLES GÉNÉRAUX:

    "Comment faire du pain ?" → STATIQUE (recette générale)
    "Quel est le prix actuel du Bitcoin ?" → DYNAMIQUE (prix changeant)
    "Qu'est-ce que la photosynthèse ?" → STATIQUE (concept scientifique)
    "Quelles sont les dernières nouvelles ?" → DYNAMIQUE (actualités récentes)
    "Comment s'inscrire à l'université ?" → STATIQUE (procédure générale)
    "Quand commence le semestre ?" → DYNAMIQUE (calendrier spécifique)
    "Date d'inscription pour nouveaux bacheliers ?" → DYNAMIQUE (échéance spécifique)
    "Deadline pour candidatures master ?" → DYNAMIQUE (date limite actuelle)
    "Qui est Einstein ?" → STATIQUE (information biographique établie)
    "Qui est le nouveau président ?" → DYNAMIQUE (changement récent)
    "Comment apprendre Python ?" → STATIQUE (méthode d'apprentissage)
    "Quel temps fait-il aujourd'hui ?" → DYNAMIQUE (conditions actuelles)
    "Histoire de la France ?" → STATIQUE (faits historiques)
    "Résultats d'élections récentes ?" → DYNAMIQUE (événements récents)

    INDICATEURS LINGUISTIQUES:

    STATIQUE: "comment", "qu'est-ce que", "pourquoi", "définition", "histoire", "général", "procédure"
    DYNAMIQUE: "récent", "nouveau", "actuel", "dernier", "maintenant", "aujourd'hui", "quand", "combien coûte", "disponible", "date de", "deadline", "échéance"

    ATTENTION SPÉCIALE - Ces questions sont TOUJOURS dynamiques:
    - Toute question demandant une DATE spécifique
    - Questions avec "quand", "date de", "deadline"
    - Questions sur les "nouveaux" étudiants/bacheliers (concerne dates actuelles)
    - Calendriers académiques, échéances d'inscription

    INSTRUCTIONS:
    1. Lis attentivement la question
    2. Cherche les indicateurs temporels et contextuels
    3. Détermine si la réponse nécessite des informations actuelles/récentes
    4. Si oui → DYNAMIQUE, si non → STATIQUE

    QUESTION: "{question}"

    Réponds UNIQUEMENT par:
    RÉPONSE: static ou dynamic

    RÉPONSE:""",

            'en': f"""You are an expert who classifies all types of questions by their temporal nature.

    SYSTEM TIME: {current_time}
    KNOWLEDGE BASE: July 1, 2025

    TASK: Determine if this question requires a STATIC or DYNAMIC response.

    DEFINITIONS:

    STATIC = Fixed, stable, factual information that doesn't change often:
    - Definitions, concepts, theories
    - Historical and geographical facts
    - General procedures and established regulations
    - Basic organizational structures
    - Established biographical information
    - Established scientific knowledge
    - General methods and techniques
    - Fundamental institutional information
    - Anything that remains true regardless of time

    DYNAMIC = Information that changes over time or requires current/recent data:
    - Current events, news, updates
    - Specific dates, schedules, calendars
    - Current statistics and data
    - Recent appointments, changes
    - Current availability of services/products
    - Prices, quotes, current markets
    - Weather, current conditions
    - "Real-time" information
    - Recent publications, releases
    - Anything requiring info after July 2025
    - Questions with temporal keywords: "recent", "new", "current", "latest", "now", "today"

    GENERAL EXAMPLES:

    "How to make bread?" → STATIC (general recipe)
    "What's the current Bitcoin price?" → DYNAMIC (changing price)
    "What is photosynthesis?" → STATIC (scientific concept)
    "What are the latest news?" → DYNAMIC (recent updates)
    "How to apply to university?" → STATIC (general procedure)
    "When does the semester start?" → DYNAMIC (specific calendar)
    "Who is Einstein?" → STATIC (established biographical info)
    "Who is the new president?" → DYNAMIC (recent change)
    "How to learn Python?" → STATIC (learning method)
    "What's the weather today?" → DYNAMIC (current conditions)
    "History of France?" → STATIC (historical facts)
    "Recent election results?" → DYNAMIC (recent events)

    LINGUISTIC INDICATORS:

    STATIC: "how", "what is", "why", "definition", "history", "general"
    DYNAMIC: "recent", "new", "current", "latest", "now", "today", "when", "how much costs", "available"

    INSTRUCTIONS:
    1. Read the question carefully
    2. Look for temporal and contextual indicators
    3. Determine if the answer requires current/recent information
    4. If yes → DYNAMIC, if no → STATIC

    QUESTION: "{question}"

    Answer ONLY with:
    ANSWER: static or dynamic

    ANSWER:""",

            'ar': f"""أنت خبير يصنف جميع أنواع الأسئلة حسب طبيعتها الزمنية.

    وقت النظام: {current_time}
    قاعدة المعرفة: 1 يوليو 2025

    المهمة: حدد ما إذا كان هذا السؤال يتطلب إجابة ثابتة أم متغيرة.

    التعريفات:

    ثابت = معلومات ثابتة ومستقرة وحقائق لا تتغير كثيراً:
    - التعريفات والمفاهيم والنظريات
    - الحقائق التاريخية والجغرافية
    - الإجراءات العامة واللوائح المؤسسة
    - الهياكل التنظيمية الأساسية
    - المعلومات السيرية المؤسسة
    - المعرفة العلمية المؤسسة
    - الطرق والتقنيات العامة
    - المعلومات المؤسسية الأساسية
    - أي شيء يبقى صحيحاً بغض النظر عن الوقت

    متغير = معلومات تتغير مع الوقت أو تتطلب بيانات حالية/حديثة:
    - الأحداث الحالية والأخبار والتحديثات
    - التواريخ والجداول والتقاويم المحددة
    - الإحصائيات والبيانات الحالية
    - التعيينات والتغييرات الأخيرة
    - التوفر الحالي للخدمات/المنتجات
    - الأسعار والعروض والأسواق الحالية
    - الطقس والظروف الحالية
    - معلومات "في الوقت الفعلي"
    - المنشورات والإصدارات الأخيرة
    - أي شيء يتطلب معلومات بعد يوليو 2025
    - أسئلة بكلمات زمنية: "حديث"، "جديد"، "حالي"، "أخير"، "الآن"، "اليوم"

    أمثلة عامة:

    "كيف أصنع الخبز؟" → ثابت (وصفة عامة)
    "ما هو سعر البيتكوين الحالي؟" → متغير (سعر متغير)
    "ما هي عملية التمثيل الضوئي؟" → ثابت (مفهوم علمي)
    "ما هي آخر الأخبار؟" → متغير (تحديثات حديثة)
    "كيف أتقدم للجامعة؟" → ثابت (إجراء عام)
    "متى يبدأ الفصل الدراسي؟" → متغير (تقويم محدد)

    مؤشرات لغوية:

    ثابت: "كيف"، "ما هو"، "لماذا"، "تعريف"، "تاريخ"، "عام"
    متغير: "حديث"، "جديد"، "حالي"، "أخير"، "الآن"، "اليوم"، "متى"، "كم يكلف"، "متوفر"

    تعليمات:
    1. اقرأ السؤال بعناية
    2. ابحث عن المؤشرات الزمنية والسياقية
    3. حدد ما إذا كانت الإجابة تتطلب معلومات حالية/حديثة
    4. إذا كانت الإجابة نعم → متغير، إذا لا → ثابت

    السؤال: "{question}"

    أجب فقط بـ:
    الإجابة: static أو dynamic

    الإجابة:""",

            'amz': f"""Anta d amussnaw i yesseflayan akk tawsitin n isqsiyen s tɣarma-nsen tazmanit.

    Akud n unagraw: {current_time}
    Taffa n tussna: 1 Yulyuz 2025

    Tanbaḍt: Ḥded ma yella asqsi-a yesra tiririt TAZṚAYT neɣ TAZGRAWANT.

    Asbadu:

    TAZṚAYT = Talɣut tazṛayt, tameqqant, n tidet ur ttinilen deg unecti:
    - Asbadu, tarma, tiẓṛiyin
    - Tidet n umezruy d trakalt
    - Tarrayin timatavin d izerfan yettwasnen
    - Tasleṭ tasdawit tagejdant
    - Talɣut n tmeddurt yettwasnen
    - Tussna tussint yettwasnen
    - Tarrayin d titiknikiyin timatavin
    - Talɣut tasdawit tagejdant
    - Yal taɣawsa i yettnayan d tidet war tiwala n wakud

    TAZGRAWANT = Talɣut ttinilen deg wakud neɣ tesra isefka imaynuten/n tura:
    - Tidyanin n tura, tisalt, ileqman
    - Azemz, iserkiyen, iwitayen yettwaheddden
    - Tiḥsayin d yisefka n tura
    - Afran d yinbeddelen imaynuten
    - Aserɣ amiran n tnbaḍt/ifarisen
    - Ssumaten, tikrayin, isuga imaynuten
    - Anezwu, addaden imaynuten
    - Talɣut "deg wakud-is"
    - Tira d yiseɣriwen imaynuten
    - Yal taɣawsa tesran talɣut deffir Yulyuz 2025
    - Isqsiyen s wawalen izmaniten: "amaynu", "n tura", "aneggaru", "tura", "ass-a"

    Imedyaten imatavin:

    "Amek ara xdemɣ aɣrum?" → TAZṚAYT (taɣect tamatavy)
    "Acḥal d ssuma n Bitcoin n tura?" → TAZGRAWANT (ssuma yettinilen)
    "D acu id fotosintiz?" → TAZṚAYT (tarma tussint)
    "D acu id isalen imaynuten?" → TAZGRAWANT (ileqman imaynuten)

    Inmal n tutlayt:

    TAZṚAYT: "amek", "d acu id", "acuɣer", "asbadu", "amezruy", "amata"
    TAZGRAWANT: "amaynu", "n tura", "aneggaru", "tura", "ass-a", "melmi", "acḥal", "yella"

    Tinaḍin:
    1. Ɣer asqsi s tsserti
    2. Nadi inmal izmaniten d imnadin
    3. Ḥded ma yella tiririt tesra talɣut n tura/tamaynut
    4. Ma yella ih → TAZGRAWANT, ma yella uhu → TAZṚAYT

    ASQSI: "{question}"

    Rrar kan s:
    TIRIRIT: static neɣ dynamic

    TIRIRIT:"""
        }
        
        try:
            prompt = classification_prompts.get(lang, classification_prompts['fr'])
            response = self._call_ollama(prompt=prompt)
            
            logger.info(f"Classification raw response: {response}")
            
            # Analyser la réponse
            response_lower = response.lower().strip()
            
            # Chercher les mots-clés de classification
            if any(keyword in response_lower for keyword in ['dynamic', 'dynamique', 'متغير', 'tazgrawant']):
                nature = "dynamic"
            elif any(keyword in response_lower for keyword in ['static', 'statique', 'ثابت', 'tazṛayt']):
                nature = "static"
            else:
                # Par défaut, considérer comme statique si pas clair
                nature = "static"
            
            logger.info(f"Question type classified as: {nature}")
            return nature
            
        except Exception as e:
            logger.error(f"Erreur lors de la classification: {str(e)}")
            return "static"  # Par défaut en cas d'erreur
    
llm_service = LLMService()

