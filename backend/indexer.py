import json
import uuid
import sys
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True
)


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

INDEX_NAME = "faq"
VECTOR_DIM = 384

def create_index():
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "question": {"type": "text"},
                        "answer": {"type": "text"},
                        "lang": {"type": "keyword"},
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
    else:
        print(f"ℹIndex '{INDEX_NAME}' already exists.")

def delete_index():
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"🗑 Index '{INDEX_NAME}' deleted.")
    else:
        print(f"ℹIndex '{INDEX_NAME}' does not exist.")

def index_faq_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = 0
    for entry in dataset:
        questions = entry.get("question", {})
        reponses = entry.get("reponse", {})
        metas = entry.get("meta", {})

        for lang in questions:
            for question in questions[lang]:
                if lang not in reponses or not reponses[lang]:
                    continue
                answer = reponses[lang][0]
                embedding = model.encode(question).tolist()

                doc = {
                    "question": question,
                    "answer": answer,
                    "lang": lang,
                    "embedding": embedding
                }

                if metas and lang in metas and metas[lang]:
                    doc["meta"] = metas[lang][0]

                client.index(index=INDEX_NAME, id=str(uuid.uuid4()), body=doc)
                total += 1

    print(f"Indexed {total} questions into OpenSearch.")

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
            print(f"❓ Unknown command: {cmd}")
            print_help()
