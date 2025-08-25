# -------------------------------  GPU ------------------------------- #

import requests
import logging
from typing import List, Dict, Any, Union
import json
import os
from datetime import datetime
import psutil
from .SerpService import get_internet_results_for_question

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
        self.model_name = "gpt-oss:latest"
        
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

    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str: #using this
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

    def is_faculty_related(self, question: str, lang: str = 'fr') -> bool: #using this
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

    def enhance_response_with_context(self, response: str, context: Dict[str, Any], lang: str = 'fr') -> str: #using this
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

    
    def _remove_duplicates(self, text: str) -> str: #using this
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

    def _remove_block_duplicates(self, text: str) -> str: #using this
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

    def simplify_question(self, question: str, lang: str = 'fr', date: datetime = None) -> list: #using this
        """
        Simplifies a complex question by extracting the main questions.
        If the question contains multiple unrelated sub-questions, separates them.
        Determines if each question is static or dynamic according to temporal criteria.
        Determines if a date in the question is greater than the reference date. if yes type = dynamic otherwise type = static.
        
        Args:
            question: Question to simplify
            lang: Language ('fr', 'en', 'ar', 'amz')
            date: Reference date for knowledge (default: datetime.now())
        
        Important:
            - Returns: List of dict with 'question', 'type', and 'reason' keys
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if date is None:
            date = datetime.now()
        
        simplification_prompts = {
            'fr': f"""Tu es un expert en analyse de questions qui simplifie les questions complexes.

            TEMPS SYSTÈME: {current_time}

            TÂCHE: 
            Analyse cette question et détermine s'il s'agit d'une question unique complexe ou de plusieurs questions distinctes.
            maximum 6 mots par question simplifiée.

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
            
            # Extract simplified questions from response
            simplified_questions = self._extract_simplified_questions(response, lang)
            logger.info(f"Simplified questions extracted: {simplified_questions}")
            
            if not simplified_questions:
                # If extraction fails, return original question
                simplified_questions = [question.strip()]
                logger.warning("No simplified questions extracted, returning original question.")
            
            # Classify each question as static or dynamic
            classified_questions = []
            for q in simplified_questions:
                classification = self._classify_question_with_temporal_logic(q, lang, date)
                logger.info(f"Classified question: {classification}")
                classified_questions.append(classification)
            
            logger.info(f"Classified questions: {classified_questions}")
            return classified_questions
            
        except Exception as e:
            logger.error(f"Error during simplification: {str(e)}")
            return [{'question': question.strip(), 'type': 'static', 'reason': 'extraction_error'}]  # Return original question in case of error

    def _extract_simplified_questions(self, response: str, lang: str = 'fr') -> list: #using this
        """
        Extracts simplified questions from LLM response
        """
        import re
        
        simplified_questions = []
        
        try:
            # First, try to extract everything after RÉSULTAT/RESULT/النتيجة/IGMAD
            result_patterns = [
                r'RÉSULTAT:\s*(.*)',
                r'résultat:\s*(.*)',
                r'RESULT:\s*(.*)',
                r'result:\s*(.*)',
                r'النتيجة:\s*(.*)',
                r'نتيجة:\s*(.*)',
                r'IGMAD:\s*(.*)',
                r'igmad:\s*(.*)'
            ]
            
            result_text = None
            for pattern in result_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    result_text = match.group(1)
                    break
            
            if result_text is None:
                result_text = response
            
            # Extract all questions from quotes in the result text
            question_matches = re.findall(r'"([^"]+)"', result_text)
            
            if question_matches:
                simplified_questions = [q.strip() for q in question_matches if q.strip()]
            else:
                # Fallback: try to extract from JSON-like arrays
                array_matches = re.findall(r'\[([^\]]+)\]', result_text)
                for array_match in array_matches:
                    # Extract quotes from each array
                    quotes_in_array = re.findall(r'"([^"]+)"', array_match)
                    simplified_questions.extend([q.strip() for q in quotes_in_array if q.strip()])
            
            # If still no questions found, try line-by-line extraction
            if not simplified_questions:
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if any(marker in line.lower() for marker in ['•', '-', '1.', '2.', '3.']) or line.endswith('?'):
                        # Clean the line
                        clean_line = re.sub(r'^[\s\-•\d\.]+', '', line).strip()
                        if clean_line and len(clean_line) > 5:
                            simplified_questions.append(clean_line)
            
            return simplified_questions[:5]  # Limit to 5 questions max
            
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            return []
    
    def _classify_question_with_temporal_logic(self, question: str, lang: str, reference_date: datetime) -> dict: #using this
        """
        Classifie une question comme statique ou dynamique avec logique temporelle avancée
        """
        import re
        from dateutil import parser
        from datetime import timedelta
        
        current_time = datetime.now()
        
        # Patterns pour extraire des dates et années
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
                # Pas d'année spécifique → probablement actuel
                return {
                    'question': question,
                    'type': 'dynamic',
                    'reason': 'Current schedule question without specific year'
                }
        
        # 4. AUTRES INDICATEURS TEMPORELS
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

    def _update_question_with_current_time(self, question: str, current_time: datetime) -> str: #using this
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

    def _format_comprehensive_qa_pairs(self, question_answer_pairs: List[Dict], lang: str) -> str: #using this
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
    
    def generate_comprehensive_response_optimized(self, original_question: str, question_answer_pairs: List[Dict], 
                                                all_documents: List[Dict], lang: str, validate_and_fallback: bool = True) -> Dict[str, Any]: #using this
        try:
            start_time = datetime.now()
            
            
            optimized_prompts = {

                'fr': {
                    'system': """Synthétise les informations pour la Faculté des Sciences d'Oujda (FSO).

                    INSTRUCTIONS:
                    1. Analyse TOUTES les questions et leurs réponses
                    2. Si les réponses ne sont PAS pertinentes pour les questions, indique "IRRELEVANT_CONTENT" au début
                    3. Génère une réponse cohérente qui traite tous les aspects
                    4. Combine les informations de différentes sources (base de données + internet)
                    5. Résous les conflits entre réponses
                    6. Indique clairement les sources d'information
                    7. Pour les informations temporelles, précise la période

                    FORMAT DE RÉPONSE:
                    - Si contenu non pertinent: commence par "IRRELEVANT_CONTENT"
                    - Sinon: donne directement la réponse finale SANS montrer ton analyse
                    - IMPORTANT: Réponds TOUJOURS en français
                    - Ne montre JAMAIS ton processus de réflexion ou d'analyse

                    CONTEXTE FSO: Faculté des Sciences Oujda, Université Mohammed Premier""",

                    'user': """QUESTION ORIGINALE: {original_question}

                    QUESTIONS ET RÉPONSES DISPONIBLES:
                    {formatted_qa_pairs}

                    meta : En cas de présence de meta n'oublier pas de les mentioner et regrouper dans une section apart.

                    GÉNÈRE une réponse comprehensive qui traite tous les aspects. Si les réponses ne sont pas pertinentes aux questions, commence par "IRRELEVANT_CONTENT". 
                    
                    IMPORTANT: Donne UNIQUEMENT la réponse finale, sans montrer ton analyse.
                    
                    si vous avez pas de reponse, dit que vous avez pas trouver des reponse.
                    """

                    
                },
                
                'en': {
                    'system': """Synthesize information for the Faculty of Sciences Oujda (FSO).

                    INSTRUCTIONS:
                    1. Analyze ALL questions and their answers
                    2. If answers are NOT relevant to questions, indicate "IRRELEVANT_CONTENT" at the beginning
                    3. Generate a coherent response addressing all aspects
                    4. Combine information from different sources (database + internet)
                    5. Resolve conflicts between answers
                    6. Clearly indicate information sources
                    7. For temporal information, specify the time period

                    RESPONSE FORMAT:
                    - If irrelevant content: start with "IRRELEVANT_CONTENT"
                    - Otherwise: give the final answer directly WITHOUT showing your analysis
                    - IMPORTANT: Always respond in English
                    - NEVER show your thinking process or analysis

                    FSO CONTEXT: Faculty of Sciences Oujda, Mohammed First University""",

                    'user': """ORIGINAL QUESTION: {original_question}

                    AVAILABLE QUESTIONS AND ANSWERS:
                    {formatted_qa_pairs}
                
                    CONTEXT: {num_questions} questions, {num_sources} sources (database + internet)
                
                    GENERATE a comprehensive response addressing all aspects. If answers are not relevant to questions, start with "IRRELEVANT_CONTENT".
                    
                    IMPORTANT: Give ONLY the final answer, without showing your analysis."""
                },

                'ar': {
                    'system': """اجمع المعلومات لكلية العلوم وجدة (FSO).

                    التعليمات:
                    1. حلل جميع الأسئلة وأجوبتها
                    2. إذا لم تكن الأجوبة ذات صلة بالأسئلة، اكتب "IRRELEVANT_CONTENT" في البداية
                    3. أنتج إجابة متماسكة تتناول جميع الجوانب
                    4. ادمج المعلومات من مصادر مختلفة (قاعدة البيانات + الإنترنت)
                    5. حل التضارب بين الأجوبة
                    6. أشر بوضوح إلى مصادر المعلومات
                    7. للمعلومات الزمنية، حدد الفترة الزمنية

                    تنسيق الإجابة:
                    - إذا كان المحتوى غير ذي صلة: ابدأ بـ "IRRELEVANT_CONTENT"
                    - وإلا: أعط الإجابة النهائية مباشرة بدون إظهار تحليلك
                    - مهم: أجب دائماً باللغة العربية
                    - لا تُظهر أبداً عملية التفكير أو التحليل

                    سياق FSO: كلية العلوم وجدة، جامعة محمد الأول""",

                    'user': """السؤال الأصلي: {original_question}

                    الأسئلة والأجوبة المتاحة:
                    {formatted_qa_pairs}

                    السياق: {num_questions} أسئلة، {num_sources} مصادر (قاعدة البيانات + الإنترنت)

                    أنتج إجابة شاملة تتناول جميع الجوانب. إذا لم تكن الأجوبة ذات صلة بالأسئلة، ابدأ بـ "IRRELEVANT_CONTENT".
                    
                    مهم: أعط الإجابة النهائية فقط، بدون إظهار تحليلك."""
                },

                'amz': {
                    'system': """Smekti tilɣutin i Tafacult n Tussniwin Ujda (FSO).

                    TISUTIN:
                    1. Ḥḍu akk tsutlin d trarranin-nsent
                    2. Ma yella trarranin ur lɣint ara i tsutlin, aru "IRRELEVANT_CONTENT" di tazwara
                    3. Skareḍ ara ara igerrez ara iteddu d akk tferkiyin
                    4. Smekti tilɣutin seg yiɣbula yemgaraden (tafka n isefka + internet)
                    5. Fessel imeɣri gar trarranin
                    6. Mel-d s wuḍiḥ iɣbula n telɣutin
                    7. I telɣutin n wakud, sbadu tawhilt

                    AMASAL N TRART:
                    - Ma yella agbur ur ilɣi ara: bdu s "IRRELEVANT_CONTENT"
                    - Ala-t: efk trart taneggaru srid ur d-teskaneḍ ara aḥḍu-nnek
                    - IMQQRAN: Err-d yal tikelt s Tamaziɣt
                    - Ur d-teskaneḍ ara abrid n iswingimen-nnek

                    AḤRIC FSO: Tafacult n Tussniwin Ujda, Tasdawit n Muḥend Amezwaru""",

                    'user': """ASUTER ANEṢLI: {original_question}

                    TSUTLIN D TRARRANIN I YELLAN:
                    {formatted_qa_pairs}

                    AḤRIC: {num_questions} tsutlin, {num_sources} iɣbula (tafka n isefka + internet)

                    SKAREḌ ara ara iggemalen ara iteddun d akk tferkiyin. Ma yella trarranin ur lɣint ara i tsutlin, bdu s "IRRELEVANT_CONTENT".
                    
                    IMQQRAN: Efk trart taneggaru kan, ur d-teskaneḍ ara aḥḍu-nnek."""
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
    
    
    def _perform_internet_fallback(self, question_answer_pairs: List[Dict], lang: str) -> str: #using this
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
                logger.info("No internet results found for fallback questions")
                return None
            
            # Generate a structured response from the internet results
            # Since get_internet_results_for_question returns a list of formatted results,
            # we need to extract and combine the answers
            combined_response = []
            
            for result in all_internet_results:
                if isinstance(result, dict) and 'answer' in result:
                    combined_response.append(f"• {result['answer']}")
            
            if combined_response:
                return "\n\n".join(combined_response)
            else:
                logger.warning("Internet results found but no valid answers extracted")
                return None
            
        except Exception as e:
            logger.error(f"Error in internet fallback: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _format_comprehensive_qa_pairs_optimized(self, question_answer_pairs: List[Dict], lang: str) -> str: #using this
        formatted_pairs = []
        
        for i, pair in enumerate(question_answer_pairs, 1):
            question = pair['question']
            documents = pair['documents']
            
            if documents:
                # Take only top 2 documents per question to reduce prompt size
                top_docs = documents[:2]
                
                # Format each document with answer, date, and meta
                doc_parts = []
                for doc in top_docs:
                    # Get the main answer (truncated if too long)
                    answer = doc.get('answer', '')
                    if len(answer) > 200:
                        answer = answer[:200] + "..."
                    
                    # Build document info with date and meta if available
                    doc_info = answer
                    
                    # Add date if available
                    if doc.get('date'):
                        doc_info += f" [Date: {doc['date']}]"
                    
                    # Add meta (links) if available
                    if doc.get('meta'):
                        doc_info += f" [Link: {doc['meta']}]"
                    
                    doc_parts.append(doc_info)
                
                answers_text = " | ".join(doc_parts)
                formatted_pair = f"Q{i}: {question}\nA{i}: {answers_text}"
            else:
                no_answer_msg = {
                    'fr': 'Aucune réponse trouvée',
                    'en': 'No answer found',
                    'ar': 'لم يتم العثور على إجابة',
                    'amz': 'Ulac trart i yettwafen'
                }.get(lang, 'Aucune réponse trouvée')
                
                formatted_pair = f"Q{i}: {question}\nA{i}: {no_answer_msg}"
            
            formatted_pairs.append(formatted_pair)
        
        return "\n\n".join(formatted_pairs)

    def _calculate_confidence_fast(self, question_answer_pairs: List[Dict], all_documents: List[Dict]) -> float: #using this
        """Fast confidence calculation without complex logic"""
        if not question_answer_pairs:
            return 0.0
        
        # Simple confidence based on coverage and document count
        questions_with_docs = len([p for p in question_answer_pairs if p['documents']])
        coverage_ratio = questions_with_docs / len(question_answer_pairs)
        
        # Average confidence from documents
        doc_confidences = [doc.get('confidence', 0.5) for doc in all_documents if 'confidence' in doc]
        avg_doc_confidence = sum(doc_confidences) / len(doc_confidences) if doc_confidences else 0.5
        
        # Combine coverage and document confidence
        final_confidence = (coverage_ratio * 0.6) + (avg_doc_confidence * 0.4)
        
        return min(final_confidence, 1.0)

    def check_question_answer_relevance(self, question: str, documents: list, lang: str) -> bool: #using this
        try:
            # Prepare documents content for checking
            documents_content = []
            for doc in documents:
                content = doc.get('content', doc.get('answer', doc.get('text', '')))
                if content:
                    documents_content.append(content[:300])  # Limit content length
            
            # Create a simple question-answer pair for relevance checking
            relevance_check_pairs = [{
                "question": question,
                "original_question": question,
                "intent": "relevance_check",
                "documents": documents,
                "type": "relevance_validation"
            }]
            
            # Use your existing generate_comprehensive_response_optimized method
            # but with a specific prompt for relevance checking
            response = self.generate_comprehensive_response_optimized(
                original_question=f"Are the provided documents relevant to answer this question: {question}? Answer only YES or NO.",
                question_answer_pairs=relevance_check_pairs,
                all_documents=documents,
                lang=lang,
                validate_and_fallback=False
            )
            
            # Extract response text
            response_text = response.get('response', '').lower().strip()
            
            # Check for relevance indicators
            if lang.lower() == 'fr':
                return any(word in response_text for word in ['oui', 'yes', 'pertinent', 'relevant', 'approprié'])
            else:
                return any(word in response_text for word in ['yes', 'relevant', 'appropriate', 'suitable'])
                
        except Exception as e:
            logger.error(f"Error checking question-answer relevance: {str(e)}")
            return True  # Default to relevant if check fails

llm_service = LLMService()
