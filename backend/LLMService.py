# -------------------------------  GPU ------------------------------- #

import requests
import logging
from typing import List, Dict, Any
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
            
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Log avant l'appel
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
                
                # Log des performances
                logger.info(f"Réponse générée en {processing_time:.2f}s")
                logger.info(f"Tokens évalués: {result.get('eval_count', 'N/A')}")
                logger.info(f"Vitesse: {result.get('eval_count', 0) / processing_time:.1f} tokens/s")
                
                return response_text
            else:
                raise Exception(f"Erreur Ollama HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de l'appel à Ollama: {str(e)}")

    def format_search_results_for_structuring(self, results: List[Dict[str, Any]]) -> str:
        """Formate tous les résultats pour permettre au LLM de les structurer"""
        if not results:
            return "Aucun résultat trouvé."
        
        formatted_results = []
        
        for i, result in enumerate(results, 1):
            formatted_result = f"""
            ═══ RÉSULTAT {i} ═══
            Question: {result.get('question', 'Non spécifiée')}
            Réponse: {result['answer']}
            Score de pertinence: {result.get('score', 'N/A')}"""
            
            if result.get('meta'):
                formatted_result += f"\nMétadonnées: {result['meta']}"
            
            formatted_results.append(formatted_result)
        
        return "\n\n".join(formatted_results)

    def generate_structured_response(self, question: str, search_results: List[Dict[str, Any]], lang: str = 'fr') -> Dict[str, Any]:
        """Génère une réponse structurée à partir de TOUS les résultats trouvés"""
        
        try:
            start_time = datetime.now()
            
            # Log du début de traitement
            logger.info(f"Génération de réponse pour: {question[:100]}...")
            
            valid_results = [r for r in search_results if r.get('answer')]
            
            if not valid_results:
                return {
                    'response': self.no_results_messages.get(lang, self.no_results_messages['fr']),
                    'confidence': 0.0,
                    'sources_used': 0,
                    'processing_time': 0,
                    'scope': 'no_results'
                }
            
            # Utiliser les prompts selon la langue
            prompt_config = self.prompts.get(lang, self.prompts['fr'])
            
            # Formatter TOUS les résultats pour le LLM
            formatted_results = self.format_search_results_for_structuring(valid_results)
            
            # Créer le prompt complet
            user_prompt = prompt_config['user'].format(
                question=question,
                search_results=formatted_results
            )
            
            # Appeler Ollama pour structurer la réponse
            structured_response = self._call_ollama(
                prompt=user_prompt,
                system_prompt=prompt_config['system']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculer la confiance basée sur le nombre et la qualité des résultats
            confidence = self._calculate_confidence(valid_results)
            
            logger.info(f"Réponse générée avec confiance: {confidence}")
            
            return {
                'response': structured_response,
                'confidence': confidence,
                'sources_used': len(valid_results),
                'processing_time': processing_time,
                'original_results': valid_results,
                'scope': 'fso_related'
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse structurée: {str(e)}")
            return {
                'response': self.no_results_messages.get(lang, self.no_results_messages['fr']),
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

    def is_faculty_related(self, question: str, lang: str = 'fr') -> Dict[str, Any]:
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
            - "NON" si la question ne concerne pas la FSO
            
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
            - "YES" if the question concerns FSO (studies, services, student life, research, administration)
            - "NO" if the question does not concern FSO
            
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
            
            Tiririt:"""
        }
        
        try:
            prompt = faculty_prompts.get(lang, faculty_prompts['fr'])
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
            
            is_related = any(keyword in response_lower for keyword in positive_keywords.get(lang, positive_keywords['fr']))
            
            return {
                'is_faculty_related': is_related,
                'confidence': 0.8 if is_related else 0.2,
                'reasoning': response,
                'lang': lang
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de pertinence: {str(e)}")
            return {
                'is_faculty_related': False,
                'confidence': 0.0,
                'reasoning': f"Erreur: {str(e)}",
                'lang': lang
            }

    def generate_faculty_response(self, question: str, lang: str = 'fr') -> Dict[str, Any]:
        """Génère une réponse basée sur les connaissances générales de la faculté"""
        
        faculty_knowledge_prompts = {
            'fr': f"""Tu es l'assistant virtuel expert de la Faculté des Sciences d'Oujda (FSO).
            
            CONNAISSANCES GÉNÉRALES FSO:
            - Faculté des Sciences de l'Université Mohammed Premier, Oujda, Maroc
            - Fondée en 1978, située sur le campus universitaire d'Oujda
            - Départements: Mathématiques-Informatique, Physique, Chimie, Biologie-Géologie
            - Formations: Licences, Masters, Doctorats
            - Environ 3000 étudiants
            - Laboratoires de recherche reconnus
            - Partenariats internationaux
            - Services: bibliothèque, laboratoires, restaurant universitaire
            - Activités: clubs scientifiques, journées portes ouvertes, colloques
            
            INSTRUCTIONS:
            1. Réponds en tant qu'assistant officiel de la FSO
            2. Utilise tes connaissances générales sur les facultés des sciences
            3. Reste factuel et professionnel
            4. Si tu n'es pas certain, indique-le clairement
            5. Encourage à contacter les services pour confirmation
            
            Question: {question}
            
            Réponse complète et structurée:""",
            
            'en': f"""You are the expert virtual assistant of the Faculty of Sciences of Oujda (FSO).
            
            GENERAL FSO KNOWLEDGE:
            - Faculty of Sciences at Mohammed Premier University, Oujda, Morocco
            - Founded in 1978, located on Oujda university campus
            - Departments: Mathematics-Computer Science, Physics, Chemistry, Biology-Geology
            - Programs: Bachelor's, Master's, PhD degrees
            - About 3000 students
            - Recognized research laboratories
            - International partnerships
            - Services: library, laboratories, university restaurant
            - Activities: scientific clubs, open days, conferences
            
            INSTRUCTIONS:
            1. Respond as the official FSO assistant
            2. Use your general knowledge about science faculties
            3. Stay factual and professional
            4. If uncertain, clearly indicate it
            5. Encourage contacting services for confirmation
            
            Question: {question}
            
            Complete and structured response:""",
            
            'ar': f"""أنت المساعد الافتراضي الخبير لكلية العلوم بوجدة (FSO).
            
            المعرفة العامة لكلية العلوم:
            - كلية العلوم بجامعة محمد الأول، وجدة، المغرب
            - تأسست عام 1978، تقع في الحرم الجامعي بوجدة
            - الأقسام: الرياضيات-الإعلاميات، الفيزياء، الكيمياء، البيولوجيا-الجيولوجيا
            - التكوينات: الإجازة، الماستر، الدكتوراه
            - حوالي 3000 طالب
            - مختبرات بحث معترف بها
            - شراكات دولية
            - الخدمات: المكتبة، المختبرات، المطعم الجامعي
            - الأنشطة: الأندية العلمية، أيام الأبواب المفتوحة، الندوات
            
            التعليمات:
            1. أجب كمساعد رسمي لكلية العلوم
            2. استخدم معرفتك العامة حول كليات العلوم
            3. ابق واقعياً ومهنياً
            4. إذا لم تكن متأكداً، أشر إلى ذلك بوضوح
            5. شجع على الاتصال بالخدمات للتأكيد
            6. أجب بالعربية
            السؤال: {question}
            
            إجابة كاملة ومنظمة:""",
            
            'amz': f"""Anta d amellal ufrawan amussnaw n tesnawalt n tussniwin n Wujda (FSO).
            
            TAMUSNI TAMATAYT FSO:
            - Tasnawalt n tussniwin n tduklit Mohammed Amezwaru, Wujda, Meṛṛuk
            - Tettwaskes deg 1978, tella deg uɣerbaz ajamɛi n Wujda
            - Igrawen: Tusnakt-Tasenṭikt, Tafizikt, Takimya, Tabyulujya-Tajyulujya
            - Iselmed: Tasɣint, Amsadur, Tibririyin
            - Azal n 3000 n yineɣmasen
            - Tinaɣin n unadi yettwasnen
            - Amɛiwen imaddunen
            - Tanbaḍt: tazeggart, tinaɣin, asekla ajamɛi
            - Tigawin: ikluban imussnanen, ussan n tewwura yeldin, inmiren
            
            IWELLIHEN:
            1. Rrar am umellal unṣib n FSO
            2. Seqdec tamusni-nnek tamatayt ɣef tesnawalt n tussniwin
            3. Ili d amaɣnu d usnakt
            4. Ma ur tettaqadeḍ ara, ini-t s tfelwit
            5. Seǧǧiɛ ad yaweḍ ɣer tanbaḍt i usentem
            
            Asqsi: {question}
            
            Tiririt teǧǧa u tettusbeḍ:"""
        }
        
        try:
            prompt = faculty_knowledge_prompts.get(lang, faculty_knowledge_prompts['fr'])
            response = self._call_ollama(prompt=prompt)
            
            return {
                'response': response,
                'confidence': 0.75, 
                'source': 'llm_general_knowledge',
                'lang': lang
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {str(e)}")
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
        """Process SERP data through LLM to generate response and structured data simultaneously"""
        
        try:
            # System prompt for processing - focus on using the EXACT client question
            system_prompt = (
                "TASK:\n"
                "1. Use the EXACT client question provided - DO NOT rephrase or change it\n"
                "2. Extract relevant facts from SERP data that answer the client question\n"
                "3. Generate clean response for client based on SERP data\n"
                "4. Create structured knowledge entry using the ORIGINAL client question\n\n"
                "RULES:\n"
                "- Use the original client question exactly as provided\n"
                "- Answer based ONLY on SERP data provided\n"
                "- If SERP data doesn't contain relevant info, say so\n"
                "- Language: " + str(lang) + "\n"
            )
            
            # Build the JSON template - emphasizing original question usage
            json_template = """{
        "user_response": "Answer to client question based on SERP data provided",
        "knowledge_entry": {
            "intent": "general_info",
            "question": {
                "fr": [
                    "THE EXACT ORIGINAL CLIENT QUESTION",
                    "variations of the original question but keeping same meaning",
                    "another variation keeping same intent"
                ],
                "en": [
                    "translation of original client question to english",
                    "english variation of original question",
                    "another english variation"
                ],
                "ar": [
                    "translation of original client question to arabic",
                    "arabic variation of original question", 
                    "another arabic variation"
                ],
                "amz": [
                    "translation of original client question to amazigh",
                    "amazigh variation of original question",
                    "another amazigh variation"
                ]
            },
            "reponse": {
                "fr": [
                    "answer in french based on SERP data",
                    "variation of answer in french",
                    "another answer variation in french"
                ],
                "en": [
                    "answer in english based on SERP data",
                    "variation of answer in english",
                    "another answer variation in english"
                ],
                "ar": [
                    "answer in arabic based on SERP data",
                    "variation of answer in arabic",
                    "another answer variation in arabic"
                ],
                "amz": [
                    "answer in amazigh based on SERP data",
                    "variation of answer in amazigh",
                    "another answer variation in amazigh"
                ]
            },
            "meta": {
                "fr": ["source links from SERP data"],
                "en": ["source links from SERP data"],
                "ar": ["source links from SERP data"],
                "amz": ["source links from SERP data"]
            }
        },
        "confidence": 0.8
    }"""
            
            # Format SERP data
            formatted_serp_data = str(serp_data)
            if isinstance(serp_data, dict):
                formatted_serp_data = self._format_context_for_llama(serp_data)
            elif isinstance(serp_data, str) and len(serp_data) > 2000:
                formatted_serp_data = serp_data[:2000] + "... [truncated]"
            
            # Build user prompt emphasizing the original question
            user_prompt = (
                "ORIGINAL CLIENT QUESTION: '" + str(question) + "'\n\n" +
                "IMPORTANT: Use this EXACT question in your JSON response. Do not rephrase it.\n\n" +
                "SERP DATA TO ANALYZE:\n" + formatted_serp_data + "\n\n" +
                "INSTRUCTIONS:\n" +
                "1. The first entry in question.fr MUST be the exact original question: '" + str(question) + "'\n" +
                "2. Answer based only on the SERP data provided\n" +
                "3. If no relevant info in SERP data, say 'No relevant information found in search results'\n\n" +
                "OUTPUT FORMAT (JSON):\n" + 
                json_template + "\n\n" +
                "Generate the JSON response now:"
            )
            
            # Call Ollama
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": self.gpu_optimized_options.copy()
            }
            
            start_time = datetime.now()
            logger.info(f"Processing SERP data for question: {question[:50]}...")
            
            response = requests.post(
                self.base_url + "/api/generate",
                json=payload,
                timeout=60000
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get('response', '').strip()
                
                logger.info("LLM response received, parsing JSON...")
                
                # Clean up the response to extract JSON
                json_start = llm_output.find('{')
                json_end = llm_output.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = llm_output[json_start:json_end]
                    processed = json.loads(json_str)
                    
                    # Verify that the original question is preserved
                    knowledge_entry = processed.get('knowledge_entry', {})
                    questions = knowledge_entry.get('question', {})
                    
                    # Force the original question to be first in French (primary language)
                    if questions.get('fr') and isinstance(questions['fr'], list):
                        # Ensure original question is first
                        if questions['fr'][0] != question:
                            questions['fr'][0] = question
                            logger.info("Corrected first French question to match original")
                    
                    logger.info("Successfully parsed LLM JSON output")
                    
                    # Store to file if requested
                    storage_success = False
                    if store_to_file and knowledge_entry:
                        storage_success = self.store_to_json_file(
                            knowledge_entry, 
                            filename
                        )
                    
                    return {
                        'display': processed.get('user_response', 'No response generated'),
                        'storage': knowledge_entry,
                        'confidence': processed.get('confidence', 0.5),
                        'stored_to_file': storage_success,
                        'file_path': filename if storage_success else None,
                        'original_question': question,  # Include for verification
                        'processing_time': processing_time
                    }
                else:
                    raise ValueError("No valid JSON found in LLM response")
                    
            else:
                raise Exception("Ollama API returned status code: " + str(response.status_code))
                
        except json.JSONDecodeError as e:
            logger.error("JSON parsing failed: " + str(e))
            logger.error("Raw LLM output: " + str(llm_output) if 'llm_output' in locals() else "No output available")
            
            # Fallback: create a basic response with original question
            fallback_entry = {
                "intent": "general_info",
                "question": {lang: [question]},
                "reponse": {lang: ["Unable to process SERP data properly"]},
                "meta": {lang: ["Processing error"]}
            }
            
            storage_success = False
            if store_to_file:
                storage_success = self.store_to_json_file(fallback_entry, filename)
            
            return {
                'display': self.no_results_messages.get(lang, self.no_results_messages['fr']),
                'storage': fallback_entry,
                'confidence': 0.0,
                'stored_to_file': storage_success,
                'file_path': filename if storage_success else None,
                'original_question': question,
                'error': 'JSON parsing failed'
            }
        
        except Exception as e:
            logger.error("Error in process_serp_to_response: " + str(e))
            
            # Fallback with original question preserved
            fallback_entry = {
                "intent": "general_info",
                "question": {lang: [question]},
                "reponse": {lang: ["Error processing request"]},
                "meta": {lang: ["System error"]}
            }
            
            storage_success = False
            if store_to_file:
                try:
                    storage_success = self.store_to_json_file(fallback_entry, filename)
                except:
                    pass
            
            return {
                'display': self.no_results_messages.get(lang, self.no_results_messages['fr']),
                'storage': fallback_entry,
                'confidence': 0.0,
                'stored_to_file': storage_success,
                'file_path': filename if storage_success else None,
                'original_question': question,
                'error': str(e)
            }

        # Helper method to read stored data    

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


# Instance globale du service
llm_service = LLMService()

