import json
import logging


logger = logging.getLogger(__name__)


def index_faq_data(dict_file, intent_val, lang, confidence):
    with open(dict_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    entry = dataset.get(intent_val)
    if not entry:
        print(f"No entry found for intent: {intent_val}")
        return []

    reponses = entry.get("reponse", {})
    metas = entry.get("meta", {})
    date = entry.get("data", str)

    if lang not in reponses:
        print(f"No responses found for lang: {lang}")
        return []

    docs = []
    for answer in reponses[lang]:
        doc = {
            "answer": answer,
            "lang": lang,
            "intent": intent_val,
            "confidence": confidence,
            "date": date
        }
        if metas and lang in metas and metas[lang]:
            doc["meta"] = metas[lang][0]

        docs.append(doc)

    return docs