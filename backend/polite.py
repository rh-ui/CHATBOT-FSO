import random
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

LANG_MAP = {
    'fr': 'fr',
    'en': 'en',
    'ar': 'ar',    
    'amz': 'amz',
}

def is_not_defined(lang: str):
    """Return a random 'not found' message based on language"""
    rnb = random.randint(0, 9)

    output = {
        "fr": [
            "Désolé, je n'ai pas trouvé de réponse pertinente.",
            "Je suis navré, je ne dispose pas de l'information demandée.",
            "Pardon, je ne peux pas répondre précisément à cette question.",
            "Excusez-moi, je n'ai pas de données à ce sujet.",
            "Je crains de ne pas pouvoir vous aider sur ce point.",
            "Malheureusement, je n'ai pas trouvé d'information adéquate.",
            "Je suis désolé, je n'ai pas pu localiser la réponse souhaitée.",
            "Veuillez m'excuser, je ne possède pas la réponse pour le moment.",
            "Navré, je ne parviens pas à trouver ce que vous cherchez.",
            "Je regrette, je ne dispose pas de renseignements à ce sujet."
        ],
        "en": [
            "Sorry, I couldn't find a relevant answer.",
            "I apologize, I don't have the requested information.",
            "Pardon me, I'm unable to answer that question right now.",
            "Unfortunately, I don't have data on this topic.",
            "I'm afraid I can't help with that at the moment.",
            "Regrettably, I couldn't locate an adequate response.",
            "Sorry, I wasn't able to find the answer you're looking for.",
            "Excuse me, I don't seem to have the information needed.",
            "I'm sorry, but I couldn't provide a precise reply.",
            "Apologies, I don't have details on that."
        ],
        "ar": [
            "عذراً، لم أتمكن من العثور على إجابة مناسبة.",
            "آسف، لا أملك المعلومات المطلوبة حالياً.",
            "أعتذر، لا أستطيع الإجابة على هذا السؤال الآن.",
            "عذراً، ليست لدي بيانات حول هذا الموضوع.",
            "أخشى أنني لا أستطيع مساعدتك في هذا الأمر حالياً.",
            "للأسف، لم أجد معلومات كافية للإجابة.",
            "آسف، لم أتمكن من الوصول إلى الجواب الذي تبحث عنه.",
            "أعتذر، لا أملك الإجابة الدقيقة في الوقت الحالي.",
            "عذراً، لم أستطع توفير رد مناسب.",
            "آسف، لا تتوفر لدي تفاصيل حول هذا السؤال."
        ],
        "amz": [
            "ⴰⵣⵓⵍ, ⵓⵔ ⵉⵜⵜⵓⴼⴽⴰ ⴰⵔⴰ ⴰⵎⵔⴰⵔⵓ ⵉⵎⵓⵔⴰⵏ.",
            "ⴰⵣⵓⵍ, ⵓⵔ ⵖⵉⵍⵖ ⴰⴷⵔⵓⵙ ⵉⵅⴼ ⵜⴻⵙⵙⵓⵜ.",
            "ⴰⵙⴽⵓⵙ, ⵓⵔ ⵏⵣⵣⵔⴰ ⴰⴷ ⴼⵖ ⴻⴷ ⵜⵉⵔⴰⵡⵉⵏ ⵉⵅⴼ.",
            "ⵉⵙⵉⵏⵏⴰⵡⵉⵏ, ⵓⵔ ⵉⵍⵍⴰ ⵢⵉⵙⴼⴽⴰ ⵉⵅⴼ ⴰⵡⵉ.",
            "ⴰⵣⵓⵍ, ⵓⵔ ⵉⵜⵜⴼⵖ ⴰⵔⴰ ⵜⵉⴷⴻⵜ ⵉ ⴷ-ⵜⵏⴼⴰ.",
            "ⴰⵣⵓⵍ, ⵓⵔ ⴷ-ⵏⵖⵉⵍ ⴰⵔⴰ ⴰⵣⴻⵜⵜⴰ ⵉⵅⴼ.",
            "ⴰⵙⴽⵓⵙ, ⵓⵔ ⴷ-ⵔⵣⵣⵓ ⴰⵔⴰ ⴰⵡⵉⴷ ⵏ ⵜⵉⵔⴰⵡⵉⵏ.",
            "ⵉⵙⵉⵏⵏⴰⵡⵉⵏ, ⵓⵔ ⵍⵍⵉⵖ ⴰⵔⴰ ⴰⵎⴻⴽ ⴰⴷ ⵔⵎⴻⴷ.",
            "ⴰⵣⵓⵍ, ⵓⵔ ⵜⴼⵓⴽ ⴰⵔⴰ ⴰⵙⵓⴼ ⵏ ⵜⵜⵉⴷⴻⵜ.",
            "ⴰⵙⴽⵓⵙ, ⵓⵔ ⵙⵙⵉⵏⴻⵖ ⴰⵔⴰ ⵜⵜⵉⴷⴻⵜ ⵏ ⵡⴰⵢⴻⵏ ⵜⵙⵏⵓⴼ."
        ]
    }

    return output.get(lang, output["fr"])[rnb]


def detect_custom_language(text: str) -> str:
    """
    Improved language detection with better handling of edge cases
    """
    if not text or not text.strip():
        return 'fr'  
    
    # Clean text
    text = text.strip()
    word_count = len(text.split())
    
    if word_count <= 2:
        if any('\u0600' <= char <= '\u06FF' for char in text):
            return 'ar'
        
        if any('\u2D30' <= char <= '\u2D7F' for char in text):
            return 'amz'
        
        french_indicators = ['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'qui', 'quelle' ]
        if text.lower() in french_indicators:
            return 'fr'
        
        english_indicators = ['the', 'and', 'or', 'is', 'are', 'was', 'were', 'you', 'me', 'my', 'your', 'his', 'her', 'our', 'their', 'how', 'when', 'ok', 'okey', 'okay']
        if text.lower() in english_indicators:
            return 'en'
        
        return 'fr'
    
    # For longer text, use langdetect
    try:
        detected_lang = detect(text)
        mapped_lang = LANG_MAP.get(detected_lang, 'fr')
        return mapped_lang
    except LangDetectException:
        # Fallback: character-based detection
        if any('\u0600' <= char <= '\u06FF' for char in text):
            return 'ar'
        elif any('\u2D30' <= char <= '\u2D7F' for char in text):
            return 'amz'
        else:
            return 'fr'

def Ibtissam_checks(question: str) -> bool:
    """
    Improved function to check if question contains FSO-related keywords
    """
    if not question or not question.strip():
        return False
    
    lang = detect_custom_language(question)
    question_lower = question.lower()

    # FSO-related keywords by language
    keywords_by_lang = {
        'fr': [
            # Core FSO terms
            "fso", "faculté des sciences", "faculté sciences", "sciences oujda", "fsoujda",
            "université mohammed premier", "ump", "oujda",
            
            # Departments
            "smi", "sma", "smc", "smp", "smg", "smb", "département",
            
            # Academic terms
            "master", "licence", "doctorat", "pfe", "stage", "inscription",
            "notes", "résultats", "soutenance", "mémoire",
            
            # General university terms
            "étudiant", "études", "cours", "examen", "diplôme", "formation",
            "professeur", "enseignant", "scolarité", "administration",
            
            # Location/contact
            "bp 717", "mohamed vi", "fso@ump.ma", "fso.ump.ma"
        ],
        
        'en': [
            "fso", "faculty of sciences", "sciences faculty", "sciences oujda",
            "mohammed premier university", "ump", "oujda",
            "department", "master", "bachelor", "doctorate", "internship",
            "student", "studies", "course", "exam", "diploma", "professor",
            "teacher", "administration", "enrollment"
        ],
        
        'ar': [
            "fso", "كلية العلوم", "علوم وجدة", "جامعة محمد الأول",
            "ump", "وجدة", "قسم", "ماستر", "إجازة", "دكتوراه",
            "طالب", "دراسة", "امتحان", "شهادة", "أستاذ", "تسجيل"
        ],
        
        'amz': [
            "fso", "taɣect n lwensa", "lwensa ujjda", "universiyt mohamed premier",
            "ump", "ujjda", "imnayen", "master", "licence", "student", "studies"
        ]
    }

    # Check keywords for the detected language
    keywords = keywords_by_lang.get(lang, keywords_by_lang['fr'])
    
    for keyword in keywords:
        if keyword.lower() in question_lower:
            return True
    
    # Additional check: if question contains university-related terms in any language
    university_terms = ["university", "université", "جامعة", "universiyt", "faculty", "faculté", "كلية", "taɣect"]
    for term in university_terms:
        if term.lower() in question_lower:
            return True
    
    return False


