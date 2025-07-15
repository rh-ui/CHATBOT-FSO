import random
from langdetect import detect


LANG_MAP = {
    'fr': 'fr',
    'en': 'en',
    'ar': 'ar',    
    'amz': 'amz',
}


def is_not_defined(lang : str) :
    rnb = random.randint(0, 9)


    output = [
        {
            "fr": [
            "Désolé, je n’ai pas trouvé de réponse pertinente.",
            "Je suis navré, je ne dispose pas de l’information demandée.",
            "Pardon, je ne peux pas répondre précisément à cette question.",
            "Excusez-moi, je n’ai pas de données à ce sujet.",
            "Je crains de ne pas pouvoir vous aider sur ce point.",
            "Malheureusement, je n’ai pas trouvé d’information adéquate.",
            "Je suis désolé, je n’ai pas pu localiser la réponse souhaitée.",
            "Veuillez m’excuser, je ne possède pas la réponse pour le moment.",
            "Navré, je ne parviens pas à trouver ce que vous cherchez.",
            "Je regrette, je ne dispose pas de renseignements à ce sujet."
            ]
        },
        {
            "en": [
            "Sorry, I couldn’t find a relevant answer.",
            "I apologize, I don’t have the requested information.",
            "Pardon me, I’m unable to answer that question right now.",
            "Unfortunately, I don’t have data on this topic.",
            "I’m afraid I can’t help with that at the moment.",
            "Regrettably, I couldn’t locate an adequate response.",
            "Sorry, I wasn’t able to find the answer you’re looking for.",
            "Excuse me, I don’t seem to have the information needed.",
            "I’m sorry, but I couldn’t provide a precise reply.",
            "Apologies, I don’t have details on that."
            ]
        },
        {
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
            ]
        },
        {
            "amz": [
            "ⴰⵣⵓⵍ, ur ttwaⵍɣ ara tirawin imara.",
            "ⴰⵣⵓⵍ, ur ɣilɣ adrus ixef tessut.",
            "ⴰⵙⴽⵓⵙ, ur nzzra ad fɣ ed tirawin ixef.",
            "ⵉⵙⵉⵏⵏⴰⵡⵉⵏ, ur illa yisefka ixf awi.",
            "ⴰⵣⵓⵍ, ur ttfɣ ara tidet i d-tnfa.",
            "ⴰⵣⵓⵍ, ur d-nɣil ara azetta ixef.",
            "ⴰⵙⴽⵓⵙ, ur d-rzzu ara awid n tirawin.",
            "ⵉⵙⵉⵏⵏⴰⵡⵉⵏ, ur lliɣ ara amek ad rmed.",
            "ⴰⵣⵓⵍ, ur tfuk ara asuf n ttidet.",
            "ⴰⵙⴽⵓⵙ, ur ssineɣ ara ttidet n wayen tsnuf."
            ]
        }
    ]


    if (lang == "fr"):
        return output[0][lang][rnb]
    elif (lang == "en"):
        return output[1][lang][rnb]
    elif (lang == "ar"):
        return output[2][lang][rnb]     
    elif (lang == "amz"):
        return output[3][lang][rnb]
    else:
        return "language is not defined, please use 'fr', 'en', 'ar' or 'amz'"
    
def detect_custom_language(text):
    word_count = len(text.split())

    if (word_count == 1):
        return 'fr' # 'fr' , 'ar', 'en', 'amz' => [ 'fr', 'ar' ... ]
    elif (word_count == 0):
        pass
    else :
            try:
                lang = detect(text)
                return LANG_MAP.get(lang, 'unknown')
            except Exception:
                return 'unknown'
    

def Ibtissam_checks(question : str) -> int : 
    """Check if the question contains specific keywords based on the language."""

    lang = detect_custom_language(question)

    

    keywords_fr = [
        # Noms officiels et abréviations
        "faculté des sciences oujda", "faculté des sciences d'oujda",
        "faculté des sciences", "fac sciences oujda", "fso", "fsoujda",
        "f.s.o", "sciences oujda", "fac sciences",
        "smi", "sma"

        # Université, localisation
        "université mohammed premier", "université mohamed premier",
        "ump oujda", "ump", "université mohammed i", "université mohamed i",
        "mohammed premier", "oujda", "hay el qods",

        # Départements et structures
        "département de biologie SMB", "département de chimie SMC",
        "département de géologie SMG", "département de physique SMP" ,
        "département d'informatique SMI", "sciences mathématiques et informatique SMA",
        "administration fso", "scolarité fso",

        # Formations & cycles
        "licence fondamentale", "licence professionnelle",
        "master", "doctorat", "pfe oujda", "stage oujda", "soutenance oujda",
        "master fso", "masters fso", "masters faculté des sciences",

        # Enseignants & direction
        "professeur oujda", "enseignant fso", "doyen faculté des sciences",
        "doyen fso", "pr.", "pr", "professeur fso",

        # Événements & activités
        "conférence", "colloque", "appel à candidatures", "candidature fso",
        "événement fso", "evenement fso",
        "conférence fso", "colloque fso", "activités scientifiques", "webinaire",
        "programme master fso", "master agroalimentaire", "master tnss",
        "master msaa", "master data science", "plateforme analytique",

        # actions :
        "inscription fso", "réinscription fso", "inscrire", "m'inscrire", "inscrit",
        "voire les notes", "notes", "FAQ", "FAQS"

        # Projets et laboratoires
        "laboratoire", "lacsA", "lacsa", "ia", "nlp team", "fso.ump.ma",
        "projet de fin d'etudes", "pfe", "plateforme analytique",

        # Qualité & gouvernance
        "management de la qualité", "gouvernance", "habilitation universitaire",

        # Dates & historique
        "crée le", "créée le", "1979", "1978", "1979", "18 avril 1979",
        "1978", "année universitaire", "année universitaire 2023", "2023-2024",
        "semestre", "s2", "s3"," s4", "s5", "s6", "s7", "s8", "s1"

        # Contacts & lieux
        "c.\u00a0p.\u00a0717", "b.p.\u00a0717", "bd mohamed vi", "bp 717",
        "+212 5365", "fso@ump.ma", "fso.ump.ma"

        # dates et événements spécifiques
        "date de création", "date de création fso",
        "date de création faculté des sciences", "date de création fso",
        "date du nomination", "date"

        # Emplois et stages
        "emploi fso", "stage fso", "offre d'emploi fso",
        "offre de stage fso", "recrutement fso", "recrutement faculté des sciences",
        "recrutement faculté des sciences oujda", "recrutement fso",
        "Emploi étudiant", "stage étudiant", "offre d'emploi étudiant",
        "offre de stage étudiant", "recrutement étudiant", "recrutement étudiant fso",

        # Autres mots-clés généraux
        "étudiant", "étudiants", "étudiante", "étudiantes",
        "études", "étudier", "étude", "recherche", "recherches",
        "recherche scientifique", "recherche et développement",
        "recherche et innovation", "innovation", "sciences",
        "sciences exactes", "sciences naturelles", "sciences appliquées",
        "sciences fondamentales", "sciences de la vie", "sciences de la terre",
        "sciences de l'environnement", "sciences de l'ingénieur",
        "sciences de l'information", "sciences de la communication",
        "sciences de l'éducation", "sciences sociales", "sciences humaines",
    ]

    keywords_en = [
        # Official names and abbreviations
        "faculty of sciences oujda", "faculty of sciences of oujda",
        "faculty of sciences", "sciences faculty oujda", "fso", "fsoujda",
        "f.s.o", "sciences oujda", "sciences faculty",

        # University, location
        "mohammed premier university", "mohammed i university",
        "ump oujda", "ump", "mohammed premier", "oujda", "hay el qods",

        # Departments & structures
        "department of biology", "department of chemistry",
        "department of geology", "department of physics",
        "department of computer science", "mathematical and computer sciences",
        "fso administration", "fso registrar",

        # Programs & cycles
        "bachelor's degree", "professional license", "master", "doctorate",
        "pfe oujda", "internship oujda", "defense oujda",
        "master fso", "fso masters", "science faculty masters",

        # Faculty & staff
        "professor oujda", "fso teacher", "dean faculty of sciences",
        "dean fso", "professor fso",

        # Events & activities
        "conference", "colloquium", "call for applications", "fso application",
        "fso enrollment", "re-enrollment fso", "fso event",
        "fso conference", "fso colloquium", "scientific activities",
        "webinar", "fso master program", "agri-food master",
        "tnss master", "msaa master", "data science master", "analytical platform",

        # Projects & labs
        "laboratory", "lacsA", "lacsa", "ai", "nlp team", "fso.ump.ma",
        "final year project", "pfe", "analytical platform",

        # Quality & governance
        "quality management", "governance", "university accreditation",

        # Dates & history
        "created on", "created on fso", "creation date",
        "1979", "1978", "april 18 1979", "academic year",
        "academic year 2023", "2023-2024", "semester",
        "s1","s2","s3","s4","s5","s6","s7","s8",

        # Contacts & locations
        "p.o. box 717", "bd mohamed vi", "+212 5365", "fso@ump.ma", "fso.ump.ma",

        # Specific dates and events
        "date of creation", "creation date fso",
        "date of creation faculty of sciences", "nomination date", "date",

        # Jobs & internships
        "fso job", "fso internship", "fso job offer", "fso internship offer",
        "fso recruitment", "faculty of sciences recruitment",
        "student job", "student internship", "student job offer",
        "student internship offer", "student recruitment", "student recruitment fso",

        # Other general keywords
        "student", "students", "study", "studies", "scientific research",
        "research and development", "innovation", "science",
        "exact sciences", "natural sciences", "applied sciences",
        "fundamental sciences", "life sciences", "earth sciences",
        "environmental sciences", "engineering sciences",
        "information sciences", "communication sciences",
        "education sciences", "social sciences", "humanities",
    ]

    keywords_amz = [
        # imen n taddarist & taqbaylit (Amazigh)
        "taɣect n lwensa ujjda", "taddart n lwensa ujjda",
        "taɣect n lwensa", "lɣect n lwensa ujjda", "fso", "fsoujda",
        "f.s.o", "lwensa ujjda", "lɣect n lwensa",

        # tamghart, tagrawla
        "universiyt mohamed premier", "ump ujjda", "ump",
        "mohamed premier", "ujjda", "hay el qods",

        # imeɣnasen
        "imnayen n biology", "imnayen n chemistry",
        "imnayen n geology", "imnayen n physics",
        "imnayen n informatique", "sciences mathématiques d informatique",
        "tanzida fso", "tanzla fso",

        # kurrus & tazwara
        "licence fondamentale", "licence professionnelle",
        "master", "haɣles", "pfe ujjda", "stage ujjda", "defense ujjda",
        "master fso", "masters fso", "masters taɣect n lwensa",

        # ulac & amadal
        "professeur ujjda", "tallen fso", "dean taɣect n lwensa",
        "dean fso", "professeur fso",

        # imuddukaln & takwrat
        "conference", "colloque", "asghim n tawjahat", "candidature fso",
        "tawjart fso", "tawjart again fso", "taqɣiḍt fso",
        "conference fso", "colloque fso", "imenuz al ilmiyya",
        "webinaire", "program master fso", "master agroalimentaire",
        "master tnss", "master msaa", "master data science",
        "analytical platform",

        # tamdint & labratwar
        "laboratoire", "lacsA", "lacsa", "ai", "nlp team", "fso.ump.ma",
        "asenḍil n tidet n tarwa", "pfe", "analytical platform",

        # tizwit & tagnawt
        "management d quality", "governance", "habilitation universitaire",

        # tassnawt & tawkayt
        "wenna 1979", "1978", "18 avril 1979", "academic year",
        "semester", "s1","s2","s3","s4","s5","s6","s7","s8",

        # iɣawen & iḍin
        "p.o. box 717", "bd mohamed vi", "+212 5365", "fso@ump.ma", "fso.ump.ma",

        # ittasnawen d takksist
        "date n talabt", "creation date fso", "nomination date", "date",

        # amuddukaln d stage
        "fso job", "fso internship", "fso job offer", "fso internship offer",
        "fso recruitment", "student job", "student internship",
        "student job offer", "student internship offer", "student recruitment fso",

        # imuddukaln agllin
        "student", "students", "study", "studies",
        "research", "innovation", "science",
        "exact sciences", "natural sciences", "applied sciences",
        "life sciences", "earth sciences", "environmental sciences",
        "engineering sciences",
        "information sciences", "communication sciences",
        "education sciences",
        "social sciences", "humanities",
    ]
    keywords_ar = [
        # الأسماء الرسمية والاختصارات
        "كلية العلوم وجدة", "كلية العلوم بوجدة", "كلية العلوم",
        "كلية علوم وجدة", "fso", "fsoujda", "f.s.o", "علوم وجدة", "كلية علوم",

        # الجامعة والموقع
        "جامعة محمد الأول", "جامعة محمد الاول", "ump وجدة", "ump",
        "جامعة محمد i", "محمد الأول", "وجدة", "حي القدس",

        # الأقسام والهياكل
        "قسم البيولوجيا", "قسم الكيمياء", "قسم الجيولوجيا", "قسم الفيزياء",
        "قسم الإعلاميات", "علوم رياضيات وإعلاميات", "إدارة fso", "مصلحة شؤون الطلبة fso",

        # التكوينات والمستويات
        "إجازة أساسية", "إجازة مهنية", "ماستر", "دكتوراه",
        "مشروع نهاية الدراسة وجدة", "تدريب وجدة", "مناقشة وجدة",
        "ماستر fso", "ماسترات fso", "ماسترات كلية العلوم",

        # الأساتذة والإدارة
        "أستاذ وجدة", "أستاذ fso", "عميد كلية العلوم", "عميد fso",
        "بروفيسور fso", "بروف.", "بروف",

        # الأنشطة والفعاليات
        "مؤتمر", "ندوة", "دعوة للترشيحات", "ترشيح fso",
        "تسجيل fso", "إعادة التسجيل fso", "حدث fso", "فعالية fso",
        "مؤتمر fso", "ندوة fso", "أنشطة علمية", "ويبينار",
        "برنامج ماستر fso", "ماستر الصناعات الغذائية", "ماستر tnss",
        "ماستر msaa", "ماستر علم البيانات", "منصة تحليلية",

        # المشاريع والمختبرات
        "مختبر", "lacsA", "lacsa", "ذكاء اصطناعي", "فريق nlp", "fso.ump.ma",
        "مشروع نهاية الدراسة", "pfe", "منصة تحليلية",

        # الجودة والحكامة
        "تسيير الجودة", "الحكامة", "الاعتماد الجامعي",

        # التواريخ والتاريخ
        "تأسست في", "تاريخ التأسيس", "1979", "1978", "18 أبريل 1979",
        "السنة الجامعية", "السنة الجامعية 2023", "2023-2024",
        "فصل دراسي", "الفصل s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8",

        # جهات الاتصال والأماكن
        "ص.ب 717", "شارع محمد السادس", "bp 717", "+212 5365", "fso@ump.ma", "fso.ump.ma",

        # التواريخ والأحداث الخاصة
        "تاريخ الإنشاء", "تاريخ إنشاء fso", "تاريخ إنشاء كلية العلوم",
        "تاريخ التعيين", "تاريخ",

        # الوظائف والتداريب
        "وظيفة fso", "تدريب fso", "عرض عمل fso", "عرض تدريب fso",
        "توظيف fso", "توظيف كلية العلوم", "توظيف كلية العلوم وجدة",
        "توظيف الطلاب", "وظيفة طالب", "تدريب طالب", "عرض عمل طالب",
        "عرض تدريب طالب", "توظيف طالب fso",

        # كلمات عامة أخرى
        "طالب", "طلاب", "طالبة", "طالبات", "دراسة", "دراسات", "يدرس", "بحث",
        "أبحاث", "بحث علمي", "بحث وتطوير", "ابتكار", "علوم",
        "العلوم الدقيقة", "العلوم الطبيعية", "العلوم التطبيقية", "العلوم الأساسية",
        "علوم الحياة", "علوم الأرض", "علوم البيئة", "علوم الهندسة",
        "علوم المعلومات", "علوم الاتصال", "علوم التربية",
        "العلوم الاجتماعية", "العلوم الإنسانية",
    ]

    if (lang == "fr"):
        for key in keywords_fr:
            if question.lower() in key.lower() or key.lower() in question.lower() or key == (keywords_fr):
                return True
    elif (lang == "en"):
        for key in keywords_en:
            if question.lower() in key.lower() or key.lower() in question.lower() or key == (keywords_en):
                return True
    elif (lang == "ar"):
        for key in keywords_ar:
            if question.lower() in key.lower() or key.lower() in question.lower() or key == (keywords_ar):
                return True
    elif (lang == "amz"):
        for key in keywords_amz:
            if question.lower() in key.lower() or key.lower() in question.lower() or key == (keywords_amz):
                return True
    else:
        return False
    

print(Ibtissam_checks('smi'))