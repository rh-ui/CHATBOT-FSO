#!/bin/bash

echo "=== DÉMARRAGE DE L'APPLICATION CHATBOT ==="

# Étape 1 : Démarrer OpenSearch (Docker)
echo
echo "[1/4] Démarrage d'OpenSearch via Docker..."
docker start opensearch 2>/dev/null || docker run -d --name opensearch -p 9200:9200 -p 9600:9600 opensearchproject/opensearch:latest

# Étape 2 : Backend : activer venv, lancer indexer.py puis uvicorn
echo
echo "[2/4] Lancement du backend : indexation et API..."

cd ~/Documents/STAGE-PFA-2/CHATBOT-FSO/backend || exit 1
source .venv/bin/activate

# Lancer indexation (en avant-plan)
python3 indexer.py reset

# Lancer uvicorn en arrière-plan
nohup uvicorn app:app --reload > uvicorn.log 2>&1 &

# Étape 3 : Frontend
echo
echo "[3/4] Lancement du frontend (UI)..."
cd ../UI || exit 1
nohup npm run dev > frontend.log 2>&1 &

echo
echo "=== TOUT EST LANCÉ ==="
