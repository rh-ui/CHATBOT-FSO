
# 🎓 M'SO - Assistant Virtuel de la Faculté des Sciences d'Oujda

<div align="center">
  <img src="https://img.shields.io/badge/React-18.2.0-blue?logo=react" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.95.2-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenSearch-2.11.1-orange?logo=opensearch" alt="OpenSearch">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</div>

---

**M'SO** est un assistant virtuel multilingue intelligent développé pour la Faculté des Sciences d'Oujda (Université Mohammed Premier).  
Il permet aux étudiants et visiteurs d'obtenir des réponses instantanées aux questions fréquentes sur la faculté, via une interface moderne et intuitive.

---

## 🌟 Fonctionnalités clés

- 🔍 Recherche intelligente (RAG) avec OpenSearch
- 🧠 Enrichissement dynamique des données
- 🌐 Multilingue : français, arabe, anglais, amazigh
- 💬 Interface moderne avec React
- ✅ Responsive, rapide et léger

---

## 🛠 Architecture



---

## ⚙️ Installation

### ✅ Prérequis

- Python 3.10+
- Node.js 16+
- Docker (pour OpenSearch)

---

### 📁 1. Backend (FastAPI)

```bash
cd back-chat
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
docker-compose up -d
python indexer.py reset
uvicorn app:app --reload
```

---

### 📁 2. Frontend (React/Vite)

```bash
cd ../front-chat
npm install
npm run dev
```

---


## 📁 Structure du projet

```
fso-chatbot/
├── back-chat/              #Backend FastAPI avec llama3:8b local
│   ├── audio_test/         # STT
│   ├── brahim_serp/        # Advanced scraper
│   ├── app.py              # application
│   ├── LLMService.py       # llama3:8b bridge
│   ├── polite.py           # language handler
│   ├── indexer.py          # Script d'indexation initiale
│   ├── dataset.json        # Base FAQ de la FSO
│   └── docker-compose.yml  # Lancement OpenSearch
│
└── front-chat/
    ├── src/
    │   ├── components/     # Composant principal du chat
    │   ├── assets/         # Logos et images
    │   ├── style/          # Fichiers CSS
    └── vite.config.ts
```

---

## 🧪 Test

- Lancer le backend sur http://localhost:8000
- Lancer le frontend sur http://localhost:5173
- L’interface est accessible depuis le coin inférieur droit
- Pose une question comme :  
  > "Quels sont les horaires d'ouverture de la bibliothèque ?"

---
 

## 🙏 Remerciements

- Université Mohammed Premier – FSO
- OpenSearch & FastAPI Community
- OpenAI – GPT API
- Contributeurs open source ❤️

---

## 📄 Licence

Ce projet est sous licence MIT.  
Voir le fichier `LICENSE` pour plus de détails.
