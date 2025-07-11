
# 🤖 Chatbot Multilingue – Backend (FSO)

Ce projet permet de poser des questions à un chatbot multilingue basé sur les données FAQ de la Faculté des Sciences d’Oujda.  
Il utilise **FastAPI**, **OpenSearch**, et **SentenceTransformers** pour le backend.

---

## 🧰 Prérequis

- ✅ Python 3.10+
- ✅ Docker + Docker Desktop (pour OpenSearch)
- ✅ (Facultatif) Postman pour tester l’API

---

## ⚙️ Étapes pour exécuter le projet

### 1️⃣ Créer un environnement virtuel et l’activer

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 2️⃣ Installer les dépendances Python

```bash
pip install fastapi uvicorn opensearch-py sentence-transformers
```

---

### 3️⃣ Lancer OpenSearch avec Docker

```bash
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 ^
  -e "discovery.type=single-node" ^
  -e "plugins.security.disabled=true" ^
  opensearchproject/opensearch:2.11.1
```

✅ **Important** : Lance **Docker Desktop** avant cette commande !

⏳ **Attendre environ 1 minute**, puis tester avec :

```bash
curl http://localhost:9200
```

Tu dois obtenir une réponse comme :

```json
{
  "name" : "440cbef57f7d",
  "cluster_name" : "docker-cluster",
  ...
  "tagline" : "The OpenSearch Project: https://opensearch.org/"
}
```

---

### 4️⃣ Indexer les données dans OpenSearch

```bash
python indexer.py
```

---

### 5️⃣ Lancer le serveur FastAPI

```bash
uvicorn app:app --reload
```

---

### 6️⃣ Tester l’API

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Quel est le numéro de téléphone principal de la faculté ?\", \"lang\": \"fr\"}"
```

> Tu peux aussi tester depuis : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📁 Structure du projet

| Fichier              | Rôle                                            |
|----------------------|-------------------------------------------------|
| `app.py`             | Serveur FastAPI pour traiter les requêtes       |
| `indexer.py`         | Script pour charger les données dans OpenSearch |
| `contact_final.json` | Données FAQ multilingues (source d'embeddings)  |

---

## 🧩 Problèmes fréquents

### ❌ Erreur de connexion à OpenSearch ?
- Vérifie que Docker Desktop est **lancé**
- Vérifie que le conteneur tourne avec :

```bash
docker ps
```

- Vérifie que le port `9200` est disponible.

---

## 🐳 Docker Compose (facultatif)

Au lieu de taper la commande docker manuellement, tu peux utiliser ce fichier :

### `docker-compose.yml`

```yaml
version: '3.7'

services:
  opensearch:
    image: opensearchproject/opensearch:2.11.1
    container_name: opensearch
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true
      - bootstrap.memory_lock=true
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
    ulimits:
      memlock:
        soft: -1
        hard: -1
    ports:
      - "9200:9200"
      - "9600:9600"
    networks:
      - opensearch-net

networks:
  opensearch-net:
    driver: bridge
```

### 🧪 Utilisation

```bash
cd back-chat
docker-compose up -d
curl http://localhost:9200
```

### 🧹 Pour arrêter proprement :

```bash
docker-compose down -v
```

---

## 📌 Remarque

Plus tard, on pourra aussi **ajouter FastAPI au `docker-compose`** pour conteneuriser tout le projet 👍

---

## ✅ FIN 🎉

Tout est prêt ! Pose ta question à `/ask` et le backend répond intelligemment 👇

```json
{
  "question": "Comment contacter la bibliothèque ?",
  "lang": "fr"
}
```

---
