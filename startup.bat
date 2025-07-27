@echo off
echo === DÉMARRAGE DE L'APPLICATION CHATBOT ===

:: Étape 1 : Démarrer OpenSearch (dans Docker)
echo.
echo > [1/4] Démarrage d'OpenSearch via Docker...
docker start opensearch || docker run -d --name opensearch -p 9200:9200 -p 9600:9600 opensearchproject/opensearch:latest

:: Étape 2 : Activer l'environnement virtuel Python , lancer l'API et reindexer faq
echo.
echo > [2/4] Lancement de l'indexation (indexer.py)...

cd C:\Users\Gigabyte\Documents\STAGE-PFA-2\CHATBOT-FSO\backend
call .venv\Scripts\activate
start cmd /k "py indexer.py reset"
echo.
echo > [3/5] Lancement du backend (FastAPI)...
start cmd /k "uvicorn app:app --reload"
cd ..

:: Étape 3 : Lancer l'interface utilisateur (UI)
echo.
echo > [3/4] Lancement du frontend (UI)...
cd C:\Users\Gigabyte\Documents\STAGE-PFA-2\CHATBOT-FSO\UI
start cmd /k "npm run dev"
cd ..

echo.
echo === TOUT EST LANCÉ ===
pause
