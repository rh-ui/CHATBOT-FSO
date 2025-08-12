"""
Script de standardisation du dataset des intents pour le classifier d'intent.
"""

import os
import csv
import re
from pathlib import Path

def detect_language(question):
    """Détection améliorée de la langue basée sur les caractères et patterns."""
    # Détection de l'arabe (caractères arabes)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
    if arabic_pattern.search(question):
        return 'arabic'
    
    # Détection de l'amazigh (caractères tifinagh)
    tifinagh_pattern = re.compile(r'[\u2D30-\u2D7F]')
    if tifinagh_pattern.search(question):
        return 'amazigh'
    
    # Détection de l'anglais (pas d'accents, caractères ASCII)
    english_pattern = re.compile(r'^[a-zA-Z\s\?\.\,\!\-\'\"]+$')
    if english_pattern.match(question) and not any(char in 'àâäéèêëïîôöùûüÿç' for char in question):
        return 'english'
    
    # Par défaut, considérer comme français
    return 'french'

def standardize_file(file_path):
    """Standardise un fichier CSV pour avoir 20 questions par langue."""
    print(f"Traitement de {file_path.name}...")
    
    # Lire le fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extraire les questions existantes par langue
    french_questions = []
    english_questions = []
    arabic_questions = []
    amazigh_questions = []
    
    for line in lines:
        if not line.strip():
            continue
            
        parts = line.split(',')
        if len(parts) < 2:
            continue
            
        question = parts[0].strip()
        intent = parts[1].strip()
        
        # Détection de la langue
        lang = detect_language(question)
        
        if lang == 'french':
            french_questions.append((question, intent))
        elif lang == 'english':
            english_questions.append((question, intent))
        elif lang == 'arabic':
            arabic_questions.append((question, intent))
        elif lang == 'amazigh':
            amazigh_questions.append((question, intent))
    
    # Générer des questions supplémentaires si nécessaire
    def generate_additional_questions(base_questions, language, count_needed, intent_name):
        if count_needed <= 0:
            return []
        
        additional = []
        base_templates = {
            'french': [
                "Pouvez-vous me donner plus d'informations sur {} ?",
                "J'aimerais en savoir plus sur {}",
                "Quels sont les détails concernant {} ?",
                "Comment fonctionne {} ?",
                "Où puis-je trouver des informations sur {} ?",
                "Quand puis-je accéder à {} ?",
                "Qui peut m'aider avec {} ?",
                "Pourquoi {} est-il important ?",
                "Combien de temps faut-il pour {} ?",
                "Quelle est la procédure pour {} ?",
                "Comment procéder pour {} ?",
                "Quelles sont les étapes pour {} ?",
                "Existe-t-il une documentation sur {} ?",
                "Peut-on avoir un exemple de {} ?",
                "Quels sont les avantages de {} ?",
                "Y a-t-il des restrictions pour {} ?",
                "Comment s'inscrire à {} ?",
                "Quels sont les prérequis pour {} ?",
                "Où se déroule {} ?",
                "Quand commence {} ?"
            ],
            'english': [
                "Can you give me more information about {}?",
                "I would like to know more about {}",
                "What are the details regarding {}?",
                "How does {} work?",
                "Where can I find information about {}?",
                "When can I access {}?",
                "Who can help me with {}?",
                "Why is {} important?",
                "How long does it take to {}?",
                "What is the procedure for {}?",
                "How to proceed with {}?",
                "What are the steps for {}?",
                "Is there documentation about {}?",
                "Can I have an example of {}?",
                "What are the benefits of {}?",
                "Are there any restrictions for {}?",
                "How to register for {}?",
                "What are the prerequisites for {}?",
                "Where does {} take place?",
                "When does {} start?",
                "What are the requirements for {}?"
            ],
            'arabic': [
                "هل يمكنك إعطائي معلومات أكثر عن {}؟",
                "أود معرفة المزيد عن {}",
                "ما هي التفاصيل المتعلقة بـ {}؟",
                "كيف يعمل {}؟",
                "أين يمكنني العثور على معلومات عن {}؟",
                "متى يمكنني الوصول إلى {}؟",
                "من يمكنه مساعدتي في {}؟",
                "لماذا {} مهم؟",
                "كم من الوقت يستغرق {}؟",
                "ما هي الإجراءات المطلوبة لـ {}؟",
                "كيف أتابع مع {}؟",
                "ما هي الخطوات المطلوبة لـ {}؟",
                "هل توجد وثائق حول {}؟",
                "هل يمكنني الحصول على مثال لـ {}؟",
                "ما هي فوائد {}؟",
                "هل توجد قيود على {}؟",
                "كيف أسجل في {}؟",
                "ما هي المتطلبات المسبقة لـ {}؟",
                "أين يقام {}؟",
                "متى يبدأ {}؟",
                "ما هي المتطلبات لـ {}؟"
            ],
            'amazigh': [
                "Tzemreḍ ad d-teɛeḍ-d ugar n isallen ɣef {}?",
                "Bɣiɣ ad ɣeṛ ugar ɣef {}",
                "D acu i d-ufrux ɣef {}?",
                "Amek i d-ixdem {}?",
                "Anida zemreɣ ad d-af isallen ɣef {}?",
                "D acu i d-ufrux amezwaru ɣef {}?",
                "Anwa i d-izmer ad d-ɛawen ɣef {}?",
                "Maɣer {} i d-amahil?",
                "ⵎⴰ ⵏⵏⴰⵢ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "Amek ara d-ssenghen {}?",
                "ⵎⴰ ⵏⵏⴰⵢ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵏⵏⴰⵢ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵏⵏⴰⵢ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵏⵏⴰⵢ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵏⵏⴰⵢ ⵉⵙⵙⵉⵔ ⵉ {}?",
                "ⵎⴰ ⵉⵙⵙⵉⵔ ⵉ {}?"
            ]
        }
        
        templates = base_templates.get(language, [])
        
        for i in range(count_needed):
            if i < len(templates):
                template = templates[i]
                question = template.format(intent_name)
            else:
                # Questions génériques supplémentaires
                if language == 'french':
                    question = f"Question supplémentaire {i+1} sur {intent_name}"
                elif language == 'english':
                    question = f"Additional question {i+1} about {intent_name}"
                elif language == 'arabic':
                    question = f"سؤال إضافي {i+1} حول {intent_name}"
                else:  # amazigh
                    question = f"Suqal amezwaru {i+1} ɣef {intent_name}"
            
            additional.append((question, intent_name))
        
        return additional
    
    # Calculer combien de questions supplémentaires sont nécessaires
    french_needed = max(0, 20 - len(french_questions))
    english_needed = max(0, 20 - len(english_questions))
    arabic_needed = max(0, 20 - len(arabic_questions))
    amazigh_needed = max(0, 20 - len(amazigh_questions))
    
    # Obtenir le nom de l'intent
    intent_name = french_questions[0][1] if french_questions else (
        english_questions[0][1] if english_questions else (
            arabic_questions[0][1] if arabic_questions else (
                amazigh_questions[0][1] if amazigh_questions else "ce service"
            )
        )
    )
    
    # Générer les questions supplémentaires
    french_questions.extend(generate_additional_questions(french_questions, 'french', french_needed, intent_name))
    english_questions.extend(generate_additional_questions(english_questions, 'english', english_needed, intent_name))
    arabic_questions.extend(generate_additional_questions(arabic_questions, 'arabic', arabic_needed, intent_name))
    amazigh_questions.extend(generate_additional_questions(amazigh_questions, 'amazigh', amazigh_needed, intent_name))
    
    # S'assurer qu'on a exactement 20 questions par langue
    french_questions = french_questions[:20]
    english_questions = english_questions[:20]
    arabic_questions = arabic_questions[:20]
    amazigh_questions = amazigh_questions[:20]
    
    # Écrire le fichier standardisé
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Écrire dans l'ordre : français, anglais, arabe, amazigh
        for question, intent in french_questions:
            writer.writerow([question, intent])
        
        for question, intent in english_questions:
            writer.writerow([question, intent])
        
        for question, intent in arabic_questions:
            writer.writerow([question, intent])
        
        for question, intent in amazigh_questions:
            writer.writerow([question, intent])
    
    print(f"  ✓ Standardisé : {len(french_questions)} français, {len(english_questions)} anglais, {len(arabic_questions)} arabe, {len(amazigh_questions)} amazigh")

def main():
    """Fonction principale qui traite tous les fichiers CSV."""
    
    intent_dir = Path("data/intent_sections")
    
    if not intent_dir.exists():
        print(f"Erreur : Le dossier {intent_dir} n'existe pas.")
        return
    
    # Trouver tous les fichiers CSV
    csv_files = list(intent_dir.glob("*.csv"))
    
    if not csv_files:
        print("Aucun fichier CSV trouvé.")
        return
    
    print(f"Trouvé {len(csv_files)} fichiers CSV à traiter.")
    print("=" * 50)
    for csv_file in csv_files:
        try:
            standardize_file(csv_file)
        except Exception as e:
            print(f"  ✗ Erreur lors du traitement de {csv_file.name}: {e}")
    
    print("=" * 50)
    print("Standardisation terminée !")

if __name__ == "__main__":
    main()