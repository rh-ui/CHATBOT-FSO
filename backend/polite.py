import random

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
