import json
import uuid
import sys
from opensearchpy import OpenSearch, logger
from sentence_transformers import SentenceTransformer


client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True
)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

INDEX_NAME = "faq"
VECTOR_DIM = 384

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
    
    
def add_single_entry(entry_data):
    """Ajoute une seule entrée à la base de données"""
    try:
        questions = entry_data.get("question", {})
        reponses = entry_data.get("reponse", {})
        metas = entry_data.get("meta", {})

        added_count = 0
        
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

                doc_id = str(uuid.uuid4())
                client.index(index=INDEX_NAME, id=doc_id, body=doc)
                added_count += 1
                
                logger.info(f"Nouvelle entrée ajoutée: {question[:50]}... (ID: {doc_id})")

        return {
            "success": True,
            "added_count": added_count,
            "message": f"Ajouté {added_count} question(s) à la base de données"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout d'entrée: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Erreur lors de l'ajout à la base de données"
        }
        

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
        elif cmd == "add":
            if len(sys.argv) < 3:
                print("Usage: python indexer.py add <json_data>")
            else:
                import json
                try:
                    entry_data = json.loads(sys.argv[2])
                    result = add_single_entry(entry_data)
                    print(result["message"])
                except json.JSONDecodeError:
                    print("Erreur: Format JSON invalide")
        else:
            print(f"Unknown command: {cmd}")
            print_help()
