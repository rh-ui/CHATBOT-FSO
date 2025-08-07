import asyncio
import os
if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


from Models.query import Query

import numpy as np
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from services.StreamGenerator import StreamGenerator

os.environ['PYTHONASYNCIODEBUG'] = '1'

LANG_MAP = {
    'fr': 'fr',
    'en': 'en',
    'ar': 'ar',    
    'amz': 'amz'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



logger = logging.getLogger(__name__)


@app.get("/")
def root():
    """Endpoint racine avec information sur l'API"""
    return {
        "message": "API Chatbot FAQ avec LLM et SERP intelligent",
        "version": "1.0",
        "endpoints": {
            "/search-stream": "Recherche FAQ avec option LLM et fallback SERP intelligent",
        },
        "features": [
            "Recherche sémantique dans la base de données",
            "Fallback intelligent vers recherche internet",
            "Filtrage et scoring des résultats SERP",
            "Extraction de snippets pertinents",
            "Intégration LLM pour structurer les réponses"
        ]
    }


"""
    COTE BACKEND : bach nrj3 loading state dynamic
        --> bghit ncreer wahd server sent event bach n9ed nchanger l'etat fl ui : 
        Etapes : 
            1) creer endpoint '/search-stream' that uses StreamingResponse    
            2) f kola etape dyal processus, envoyer un evenement avec le statut actuel
            3) Utiliser yield pour envoyer les messages de statut progressivement
            
        --> apres, khassni n restaurer l flux de traitement (avant chaque etape envoyer un event SSE m3ah l msg li bit)
            exple : "Je vérifie si votre question concerne la FSO...", "Recherche sur le web en cours...", "Structuration de la réponse avec l'IA...", "Validation de la pertinence..."
            
        --> Format des events SSE : object JSON avec type[status, final] et message, Les messages suivent le format data: {json}\n\n
            < yield f"data: {json.dumps({'type': '.....', 'message': '....'})}\n\n">
        --> last event SSE howa li fih la rep finale bayna !
"""

@app.post("/search-stream")
def search_stream(query: Query) :
    generator = StreamGenerator(query)
    return StreamingResponse(
        generator.stream_search(),
        media_type ="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )



