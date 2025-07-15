import openai
import logging
from typing import List, Dict, Any
from pydantic import BaseModel
import json
import os
from datetime import datetime
from polite import is_not_defined

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Configuration pour GitHub Models
        self.client = openai.OpenAI(
            base_url="https://models.github.ai/inference",
            api_key="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"
        )
        
        self.model_name = "openai/gpt-4.1" 
        
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
           
            valid_results = [r for r in search_results if r.get('answer')]
            
            if not valid_results:
                return {
                    'response': is_not_defined(lang),
                    'confidence': 0.0,
                    'sources_used': 0,
                    'processing_time': 0,
                    'scope': 'no_results'
                }
            
            # Utiliser les prompts selon la langue
            prompt_config = self.prompts.get(lang, self.prompts['fr'])
            
            # Formatter TOUS les résultats pour le LLM
            formatted_results = self.format_search_results_for_structuring(valid_results)
            
            # Créer les messages pour l'API
            messages = [
                {"role": "system", "content": prompt_config['system']},
                {"role": "user", "content": prompt_config['user'].format(
                    question=question,
                    search_results=formatted_results
                )}
            ]
            
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1200,
                temperature=0.1,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.2
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            

            structured_response = response.choices[0].message.content.strip()
            
            # Calculer la confiance basée sur le nombre et la qualité des résultats
            confidence = self._calculate_confidence(valid_results)
            
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
                'response': is_not_defined(lang),
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
        
        
        scores = []
        for r in results:
            score = r.get('score', 0)
            if score > 1.0:
                score = score / 100
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        
        # Bonus pour plusieurs sources
        source_bonus = min(len(results) * 0.05, 0.2)
        
        # Bonus si les scores sont élevés
        high_score_bonus = 0.1 if avg_score > 0.8 else 0.0
        
        final_confidence = min(avg_score + source_bonus + high_score_bonus, 1.0)
        
        return round(final_confidence, 2)

    def enhance_response_with_context(self, response: str, context: Dict[str, Any], lang: str = 'fr') -> str:
        """Améliore la réponse avec du contexte supplémentaire"""
        try:
            context_prompts = {
                'fr': f"""Tu es l'assistant de la FSO. Améliore cette réponse en ajoutant du contexte pertinent SANS changer les informations factuelles:

                        Réponse actuelle:
                        {response}

                        Contexte supplémentaire:
                        {json.dumps(context, ensure_ascii=False, indent=2)}

                        Améliore la réponse en intégrant naturellement le contexte, mais garde toutes les informations factuelles intactes.""",
                                        
                                        'en': f"""You are the FSO assistant. Enhance this response by adding relevant context WITHOUT changing factual information:

                                        Current response:
                                        {response}

                                        Additional context:
                                        {json.dumps(context, ensure_ascii=False, indent=2)}

                                        Enhance the response by naturally integrating the context, but keep all factual information intact.""",
                                                        
                                        'ar': f"""أنت مساعد كلية العلوم. حسّن هذه الإجابة بإضافة السياق المناسب دون تغيير المعلومات الواقعية:

                                        الإجابة الحالية:
                                        {response}

                                        السياق الإضافي:
                                        {json.dumps(context, ensure_ascii=False, indent=2)}

                                        حسّن الإجابة بدمج السياق بطريقة طبيعية، ولكن احتفظ بجميع المعلومات الواقعية سليمة.""",
                                                        
                                        'amz': f"""Anta d amellal n tesnawalt. Seǧhed tiririt-a s tmerna n umnaḍ ifaqen ur d-tbeddel ara tilɣa n tidet:

                                        Tiririt n tura:
                                        {response}

                                        Amnaḍ nniḍen:
                                        {json.dumps(context, ensure_ascii=False, indent=2)}

                                        Seǧhed tiririt s usdukkel n umnaḍ s tarrayt tagamant, maca eǧǧ akk tilɣa n tidet."""
            }
            
            messages = [
                {"role": "user", "content": context_prompts.get(lang, context_prompts['fr'])}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1400,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'amélioration avec contexte: {str(e)}")
            return response

llm_service = LLMService()
