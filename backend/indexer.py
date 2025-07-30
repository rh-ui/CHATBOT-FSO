import json
import uuid
import sys
from opensearchpy import OpenSearch, logger
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True
)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

INDEX_NAME = "faq"
VECTOR_DIM = 768

def print_help():
    print("""
        Usage: python indexer.py [command]

        Commands:
          create       Create index (if not exists)
          index        Index data from dataset.json
          reset        Delete index, recreate it, and index data
          delete       Delete the index
          help         Show this help message
    """)

def create_index():
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "multilingual_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "stemmer"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "question": {
                            "type": "text",
                            "analyzer": "multilingual_analyzer",
                            "fields": {"raw": {"type": "keyword"}}
                        },
                        "answer": {"type": "text"},
                        "lang": {"type": "keyword"},
                        "intent": {"type": "keyword"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": VECTOR_DIM
                        },
                        "meta": {"type": "text"}  
                    }
                }
            }
        )
        print(f"Index '{INDEX_NAME}' created.")

def delete_index():
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"🗑 Index '{INDEX_NAME}' deleted.")
    else:
        print(f"ℹIndex '{INDEX_NAME}' does not exist.")

def index_faq_data(dict_file, intent_val, lang, confidence):
    # Load the dict structure
    with open(dict_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Get the exact entry in O(1)
    entry = dataset.get(intent_val)
    if not entry:
        print(f"No entry found for intent: {intent_val}")
        return []

    reponses = entry.get("reponse", {})
    metas = entry.get("meta", {})  # Some entries may not have this

    if lang not in reponses:
        print(f"No responses found for lang: {lang}")
        return []

    docs = []
    for answer in reponses[lang]:
        doc = {
            "answer": answer,
            "lang": lang,
            "intent": intent_val,
            "confidence": confidence
        }
        if metas and lang in metas and metas[lang]:
            doc["meta"] = metas[lang][0]

        docs.append(doc)

    return docs
     

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
    else:
        cmd = sys.argv[1].lower()
        if cmd == "create":
            create_index()
        elif cmd == "index":
            index_faq_data("dataset.json")
        elif cmd == "reset":
            delete_index()
            create_index()
            index_faq_data("dataset.json")
        elif cmd == "delete":
            delete_index()
        elif cmd == "help":
            print_help()

        else:
            print(f"Unknown command: {cmd}")
            print_help()