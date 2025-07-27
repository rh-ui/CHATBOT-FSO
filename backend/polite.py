import random
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
from typing import Dict, List, Set
from fuzzywuzzy import fuzz
import unicodedata

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

    return output.get(lang, output[lang])[rnb]


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



def normalize_text(text: str) -> str:
    """Normalise le texte en supprimant les accents et caractères spéciaux"""
    # Supprimer les accents
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    # Nettoyer les caractères spéciaux mais garder les espaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normaliser les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_meaningful_words(text: str, lang: str) -> Set[str]:
    """Extrait les mots significatifs en filtrant les mots vides"""
    stopwords = {
        'fr': {'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'je', 'tu', 
               'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'qui', 'que', 'quoi', 'comment',
               'pourquoi', 'quand', 'où', 'est', 'sont', 'avoir', 'être', 'faire', 'aller',
               'venir', 'voir', 'savoir', 'pouvoir', 'vouloir', 'devoir', 'falloir', 'dans',
               'sur', 'avec', 'pour', 'par', 'sans', 'sous', 'entre', 'pendant', 'avant',
               'après', 'depuis', 'jusqu', 'vers', 'chez', 'moi', 'toi', 'lui', 'eux', 'se'},
        'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
               'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
               'after', 'above', 'below', 'between', 'among', 'i', 'you', 'he', 'she', 'it',
               'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
               'its', 'our', 'their', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
               'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'},
        'ar': {'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'تحت', 'فوق', 'بين',
               'أن', 'إن', 'كان', 'كانت', 'يكون', 'تكون', 'هو', 'هي', 'هم', 'هن', 'أنا',
               'أنت', 'نحن', 'لي', 'لك', 'له', 'لها', 'لنا', 'لهم', 'هذا', 'هذه', 'ذلك',
               'تلك', 'التي', 'الذي', 'ما', 'كيف', 'متى', 'أين', 'لماذا', 'ماذا'},
        'amz': {'n', 'i', 'd', 'ar', 'as', 'ay', 'ur', 's', 'nni', 'nns', 'nna', 'nnk'}
    }
    
    words = set(normalize_text(text).split())
    lang_stopwords = stopwords.get(lang, set())
    
    # Filtrer les mots vides et garder seulement les mots de plus de 2 caractères
    meaningful_words = {word for word in words 
                       if len(word) > 2 and word not in lang_stopwords}
    
    return meaningful_words

def calculate_context_score(text: str, lang: str) -> float:
    """Calcule un score basé sur le contexte universitaire/académique"""
    academic_patterns = {
        'fr': [
            r'\b(étudier|étudiant|étudiante|étude)\b',
            r'\b(cours|matière|module|unité)\b',
            r'\b(examen|contrôle|test|évaluation)\b',
            r'\b(inscription|s\'inscrire|candidature)\b',
            r'\b(diplôme|certificat|attestation)\b',
            r'\b(semestre|trimestre|année universitaire)\b',
            r'\b(faculté|université|établissement)\b',
            r'\b(master|licence|doctorat|bac)\b',
            r'\b(professeur|enseignant|formateur)\b',
            r'\b(stage|projet|mémoire|thèse)\b'
        ],
        'en': [
            r'\b(study|student|studies|studying)\b',
            r'\b(course|subject|module|unit)\b',
            r'\b(exam|test|evaluation|assessment)\b',
            r'\b(enrollment|enroll|application|apply)\b',
            r'\b(diploma|certificate|degree)\b',
            r'\b(semester|quarter|academic year)\b',
            r'\b(faculty|university|college|school)\b',
            r'\b(master|bachelor|doctorate|phd)\b',
            r'\b(professor|teacher|instructor)\b',
            r'\b(internship|project|thesis|dissertation)\b'
        ],
        'ar': [
            r'\b(طالب|طلاب|دراسة|يدرس)\b',
            r'\b(مقرر|مادة|وحدة|كورس)\b',
            r'\b(امتحان|اختبار|تقييم)\b',
            r'\b(تسجيل|يسجل|طلب|يطلب)\b',
            r'\b(شهادة|دبلوم|إجازة)\b',
            r'\b(فصل|ترم|سنة دراسية)\b',
            r'\b(كلية|جامعة|مؤسسة)\b',
            r'\b(ماجستير|بكالوريوس|دكتوراه)\b',
            r'\b(أستاذ|مدرس|معلم)\b',
            r'\b(تدريب|مشروع|رسالة|أطروحة)\b'
        ]
    }
    
    patterns = academic_patterns.get(lang, [])
    score = 0
    normalized_text = normalize_text(text)
    
    for pattern in patterns:
        if re.search(pattern, normalized_text, re.IGNORECASE):
            score += 1
    
    return min(score / len(patterns), 1.0) if patterns else 0

def fuzzy_match_keywords(text: str, keywords: List[str], threshold: int = 80) -> bool:
    """Recherche floue des mots-clés avec seuil de similarité"""
    normalized_text = normalize_text(text)
    
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        
        # Vérification exacte
        if normalized_keyword in normalized_text:
            return True
        
        # Vérification floue pour les mots-clés importants (plus de 4 caractères)
        if len(normalized_keyword) > 4:
            # Diviser le texte en segments pour la comparaison
            words = normalized_text.split()
            for i in range(len(words)):
                for j in range(i + 1, min(i + len(normalized_keyword.split()) + 2, len(words) + 1)):
                    segment = ' '.join(words[i:j])
                    if fuzz.ratio(normalized_keyword, segment) >= threshold:
                        return True
    
    return False

def detect_academic_questions(text: str, lang: str) -> float:
    """
    Détecte les questions académiques typiques même sans mots-clés FSO explicites
    """
    academic_question_patterns = {
        'fr': [
            # Questions administratives
            (r'\b(qui est|c\'est qui|c est qui).*(doyen|directeur|responsable|chef)\b', 8),
            (r'\b(où|comment).*(voir|consulter|accéder|trouver).*(notes|résultats|relevés)\b', 7),
            (r'\b(comment).*(s\'inscrire|inscription|candidature|postuler)\b', 7),
            (r'\b(quand|date).*(inscription|rentrée|examens|soutenances)\b', 6),
            (r'\b(où|comment).*(payer|payement|frais).*(inscription|scolarité)\b', 7),
            
            # Questions sur les programmes
            (r'\b(quels?|quelles?).*(filières?|spécialités?|masters?|formations?)\b', 6),
            (r'\b(comment).*(choisir|sélectionner).*(filière|spécialité|option)\b', 6),
            (r'\b(conditions?).*(admission|accès).*(master|licence|doctorat)\b', 7),
            
            # Questions sur les cours
            (r'\b(emploi du temps|planning|programme|syllabus)\b', 6),
            (r'\b(qui|quel).*(professeur|enseignant|prof)\b', 5),
            (r'\b(salle|amphithéâtre|lieu).*(cours|examen|soutenance)\b', 5),
            
            # Questions sur les procédures
            (r'\b(comment).*(rattraper|repasser).*(examen|matière|module)\b', 6),
            (r'\b(stage|pfe|projet de fin).*(comment|où|quand)\b', 7),
            (r'\b(documents?|pièces?).*(fournir|nécessaires?|demandés?)\b', 6),
            
            # Questions générales universitaires
            (r'\b(bibliothèque|labo|laboratoire).*(horaires?|accès|où)\b', 5),
            (r'\b(bourse|aide financière|financement)\b', 5),
            (r'\b(transport|résidence|logement).*(étudiants?)\b', 4),
            
            # Mots-clés académiques seuls dans une question
            (r'^\s*(doyen|directeur|secrétariat|administration)\s*\??\s*$', 6),
            (r'^\s*(notes|résultats|relevés?)\s*\??\s*$', 6),
            (r'^\s*(inscription|candidature)\s*\??\s*$', 6),
            (r'^\s*(emploi du temps|planning)\s*\??\s*$', 5),
        ],
        
        'en': [
            # Administrative questions
            (r'\b(who is|who\'s).*(dean|director|head|responsible)\b', 8),
            (r'\b(where|how).*(see|check|access|find).*(grades|results|marks)\b', 7),
            (r'\b(how).*(enroll|register|apply|admission)\b', 7),
            (r'\b(when|date).*(registration|enrollment|exams|defense)\b', 6),
            (r'\b(where|how).*(pay|payment|fees).*(tuition|registration)\b', 7),
            
            # Program questions
            (r'\b(what|which).*(programs?|majors?|specializations?|degrees?)\b', 6),
            (r'\b(how).*(choose|select).*(major|specialization|program)\b', 6),
            (r'\b(requirements?).*(admission|entry).*(master|bachelor|phd)\b', 7),
            
            # Course questions
            (r'\b(schedule|timetable|syllabus|curriculum)\b', 6),
            (r'\b(who|which).*(professor|teacher|instructor)\b', 5),
            (r'\b(room|classroom|hall).*(class|exam|defense)\b', 5),
            
            # Procedure questions
            (r'\b(how).*(retake|repeat).*(exam|course|subject)\b', 6),
            (r'\b(internship|thesis|project).*(how|where|when)\b', 7),
            (r'\b(documents?|papers?).*(required|needed|necessary)\b', 6),
            
            # General university questions
            (r'\b(library|lab|laboratory).*(hours?|access|where)\b', 5),
            (r'\b(scholarship|financial aid|funding)\b', 5),
            (r'\b(transport|residence|housing).*(students?)\b', 4),
            
            # Single academic keywords in questions
            (r'^\s*(dean|director|secretary|administration)\s*\??\s*$', 6),
            (r'^\s*(grades|results|marks)\s*\??\s*$', 6),
            (r'^\s*(registration|enrollment)\s*\??\s*$', 6),
            (r'^\s*(schedule|timetable)\s*\??\s*$', 5),
        ],
        
        'ar': [
            # أسئلة إدارية
            (r'\b(من هو|من).*(عميد|مدير|رئيس|مسؤول)\b', 8),
            (r'\b(أين|كيف).*(أرى|أشاهد|أصل|أجد).*(درجات|نتائج|علامات)\b', 7),
            (r'\b(كيف).*(أسجل|تسجيل|أتقدم|طلب)\b', 7),
            (r'\b(متى|تاريخ).*(تسجيل|قبول|امتحانات|مناقشة)\b', 6),
            (r'\b(أين|كيف).*(أدفع|دفع|رسوم).*(تسجيل|دراسة)\b', 7),
            
            # أسئلة البرامج
            (r'\b(ما هي|أي).*(تخصصات|برامج|ماجستير|تكوينات)\b', 6),
            (r'\b(كيف).*(أختار|اختيار|أحدد).*(تخصص|فرع|خيار)\b', 6),
            (r'\b(شروط).*(قبول|دخول).*(ماجستير|إجازة|دكتوراه)\b', 7),
            
            # أسئلة الدروس
            (r'\b(جدول|برنامج|منهج)\b', 6),
            (r'\b(من|أي).*(أستاذ|مدرس|معلم)\b', 5),
            (r'\b(قاعة|مدرج|مكان).*(درس|امتحان|مناقشة)\b', 5),
            
            # أسئلة الإجراءات
            (r'\b(كيف).*(أعيد|إعادة).*(امتحان|مادة|وحدة)\b', 6),
            (r'\b(تدريب|مشروع تخرج|أطروحة).*(كيف|أين|متى)\b', 7),
            (r'\b(وثائق|أوراق).*(مطلوبة|ضرورية|مطلوب)\b', 6),
            
            # أسئلة جامعية عامة
            (r'\b(مكتبة|مختبر).*(ساعات|دخول|أين)\b', 5),
            (r'\b(منحة|مساعدة مالية|تمويل)\b', 5),
            (r'\b(نقل|سكن|إقامة).*(طلاب)\b', 4),
            
            # كلمات أكاديمية منفردة في أسئلة
            (r'^\s*(عميد|مدير|سكرتارية|إدارة)\s*\??\s*$', 6),
            (r'^\s*(نتائج|درجات|علامات)\s*\??\s*$', 6),
            (r'^\s*(تسجيل|قبول)\s*\??\s*$', 6),
            (r'^\s*(جدول|برنامج)\s*\??\s*$', 5),
        ],
        
        'amz': [
            # Questions académiques berbères
            (r'\b(ma d|anwa).*(dean|amqqran|amesbah)\b', 8),
            (r'\b(anida|mamek).*(ad zeɣ|ad ẓẓar).*(tibratin|igmaḍ)\b', 7),
            (r'\b(mamek).*(ad sqedceɣ|asqedec)\b', 7),
            (r'\b(melmi|ass).*(asqedec|iseɣta|imtihan)\b', 6),
            
            # Mots-clés académiques seuls
            (r'^\s*(dean|amqqran|tibratin|igmaḍ)\s*\??\s*$', 6),
        ]
    }
    
    patterns = academic_question_patterns.get(lang, [])
    max_score = 0
    normalized_text = normalize_text(text)
    
    for pattern, score in patterns:
        if re.search(pattern, normalized_text, re.IGNORECASE):
            max_score = max(max_score, score)
    
    return max_score

def check_question_intent(text: str, lang: str) -> bool:
    """Détermine si le texte ressemble à une question universitaire/académique"""
    question_indicators = {
        'fr': [
            r'\b(comment|pourquoi|quand|où|quel|quelle|quels|quelles|qui|quoi)\b',
            r'\b(peux-tu|pouvez-vous|est-ce que|qu\'est-ce que)\b',
            r'\b(aide|aider|information|renseignement|conseil)\b',
            r'\b(procédure|démarche|étapes|processus)\b',
            r'\?(.*)',  # Texte se terminant par ?
        ],
        'en': [
            r'\b(how|why|when|where|what|which|who|whom)\b',
            r'\b(can you|could you|would you|do you|is it|are there)\b',
            r'\b(help|information|advice|guidance)\b',
            r'\b(procedure|process|steps|way)\b',
            r'\?(.*)',
        ],
        'ar': [
            r'\b(كيف|لماذا|متى|أين|ما|ماذا|من|أي)\b',
            r'\b(هل|أهل|يمكن|تستطيع|تقدر)\b',
            r'\b(مساعدة|معلومات|نصيحة|إرشاد)\b',
            r'\b(إجراء|خطوات|طريقة|عملية)\b',
            r'\؟(.*)',
        ]
    }
    
    indicators = question_indicators.get(lang, [])
    normalized_text = normalize_text(text)
    
    for pattern in indicators:
        if re.search(pattern, normalized_text, re.IGNORECASE):
            return True
    
    return False

def Ibtissam_checks(question: str, lang: str) -> bool:

    if not question or not question.strip():
        return False
    
    if len(question.strip()) < 3:  # Questions trop courtes
        return False
    
    question_lower = question.lower()
    
    # Mots-clés FSO spécifiques avec pondération
    core_keywords = {
        'fr': {
            'high_priority': [
                "fso", "faculté des sciences", "faculté sciences", "sciences oujda", 
                "fsoujda", "université mohammed premier", "ump oujda",
                "fso@ump.ma", "fso.ump.ma"
            ],
            'medium_priority': [
                "oujda", "mohammed premier", "ump", "bp 717", "mohamed vi",
                "smi", "sma", "smc", "smp", "smg", "smb"
            ],
            'context_keywords': [
                "inscription", "master", "licence", "doctorat", "pfe", "stage",
                "notes", "résultats", "soutenance", "mémoire", "département",
                "étudiant", "études", "cours", "examen", "diplôme", "formation",
                "professeur", "enseignant", "scolarité", "administration"
            ]
        },
        'en': {
            'high_priority': [
                "fso", "faculty of sciences", "sciences faculty", "sciences oujda",
                "mohammed premier university", "ump oujda"
            ],
            'medium_priority': [
                "oujda", "mohammed premier", "ump"
            ],
            'context_keywords': [
                "enrollment", "master", "bachelor", "doctorate", "internship",
                "department", "student", "studies", "course", "exam", "diploma",
                "professor", "teacher", "administration"
            ]
        },
        'ar': {
            'high_priority': [
                "fso", "كلية العلوم", "علوم وجدة", "جامعة محمد الأول"
            ],
            'medium_priority': [
                "وجدة", "محمد الأول", "ump"
            ],
            'context_keywords': [
                "تسجيل", "ماستر", "إجازة", "دكتوراه", "قسم",
                "طالب", "دراسة", "امتحان", "شهادة", "أستاذ"
            ]
        },
        'amz': {
            'high_priority': [
                "fso", "taɣect n lwensa", "lwensa ujjda", "universiyt mohamed premier"
            ],
            'medium_priority': [
                "ujjda", "mohamed premier", "ump"
            ],
            'context_keywords': [
                "imnayen", "master", "licence", "student", "studies"
            ]
        }
    }
    
    if lang not in core_keywords:
        lang = 'fr'  # Fallback vers français
    
    keywords_dict = core_keywords[lang]
    
    # Score système
    score = 0
    
    # 1. Vérification des mots-clés haute priorité (score élevé)
    if fuzzy_match_keywords(question, keywords_dict['high_priority'], threshold=85):
        score += 10
    
    # 2. Vérification des mots-clés moyenne priorité (score moyen)
    if fuzzy_match_keywords(question, keywords_dict['medium_priority'], threshold=80):
        score += 5
    
    # 3. Vérification des mots-clés de contexte (score faible)
    context_matches = sum(1 for kw in keywords_dict['context_keywords'] 
                         if normalize_text(kw) in normalize_text(question))
    score += min(context_matches * 2, 6)  # Maximum 6 points pour le contexte
    
    # 4. Score de contexte académique
    academic_score = calculate_context_score(question, lang)
    score += academic_score * 4
    
    # 5. Détection de questions académiques courantes
    academic_questions_score = detect_academic_questions(question, lang)
    score += academic_questions_score
    
    # 6. Bonus si c'est une vraie question
    if check_question_intent(question, lang):
        score += 2
    
    # 7. Vérification des termes universitaires génériques dans toutes les langues
    universal_terms = [
        "university", "université", "جامعة", "universiyt",
        "faculty", "faculté", "كلية", "taɣect",
        "science", "sciences", "علوم", "lwensa"
    ]
    
    if fuzzy_match_keywords(question, universal_terms, threshold=85):
        score += 3
    
    # 8. Pénalité pour les questions trop génériques
    generic_patterns = [
        r'^(bonjour|salut|hello|hi|السلام|azul)$',
        r'^(oui|non|yes|no|نعم|لا|ih|uhu)$',
        r'^(merci|thanks|شكرا|tanmirt)$'
    ]
    
    for pattern in generic_patterns:
        if re.match(pattern, normalize_text(question)):
            score -= 5
    
    # 9. Analyse de la longueur et complexité
    words = extract_meaningful_words(question, lang)
    if len(words) >= 3:  # Au moins 3 mots significatifs
        score += 1
    
    # Seuil de décision intelligent adaptatif
    min_score = 6  # Seuil de base réduit pour capter plus de questions académiques
    
    # Ajustement du seuil selon la longueur
    if len(question.split()) < 4:
        min_score = 5   # Seuil plus bas pour les questions courtes comme "Qui est le doyen?"
    elif len(question.split()) > 15:
        min_score = 8   # Seuil plus élevé pour les questions très longues
    
    # Bonus pour les questions avec motifs académiques forts
    if academic_questions_score >= 6:
        min_score = min(min_score, 4)  # Réduire le seuil si forte indication académique
    
    return score >= min_score