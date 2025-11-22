# 🤖 Chatbot FSO - Assistant Virtuel Intelligent

<div align="center">

![FSO Logo](UI/public/logo_fso.jpeg)

**Assistant virtuel officiel de la Faculté des Sciences d'Oujda (FSO)**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.1.0-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8.3-3178C6?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4.1.11-38B2AC?style=for-the-badge&logo=tailwind-css)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

</div>

## 📋 Table des Matières

- [🎯 Vue d'ensemble](#-vue-densemble)
- [✨ Fonctionnalités](#-fonctionnalités)
- [🏗️ Architecture](#️-architecture)
- [🚀 Installation](#-installation)
- [💻 Utilisation](#-utilisation)
- [🔧 Configuration](#-configuration)
- [📁 Structure du Projet](#-structure-du-projet)
- [🤝 Contribution](#-contribution)
- [📄 Licence](#-licence)

## 🎯 Vue d'ensemble

Le **Chatbot FSO** est un assistant virtuel intelligent développé pour la Faculté des Sciences d'Oujda. Il combine des technologies de pointe en IA, traitement du langage naturel et recherche web pour fournir des réponses précises et contextuelles aux questions des étudiants et visiteurs.

### 🎓 Objectifs

- **Assistance 24/7** : Réponses instantanées aux questions fréquentes
- **Intelligence Artificielle** : Compréhension contextuelle et réponses structurées
- **Intégration FSO** : Base de connaissances spécialisée pour l'établissement
- **Interface Moderne** : Expérience utilisateur intuitive et responsive
- **Multilingue** : Support français et anglais

## ✨ Fonctionnalités

### 🧠 Intelligence Artificielle
- **Classification d'intentions** : Reconnaissance automatique du type de question
- **Traitement multilingue** : Support français et anglais avec détection automatique
- **Structuration intelligente** : Organisation logique des réponses par l'IA
- **Fallback intelligent** : Recherche web automatique en cas d'absence de données locales

### 🔍 Recherche et Base de Données
- **Recherche sémantique** : Compréhension du sens des questions
- **Base de connaissances FSO** : Données spécialisées de l'établissement
- **Intégration SERP** : Recherche internet intelligente avec filtrage
- **Scoring des résultats** : Évaluation automatique de la pertinence

### 🎨 Interface Utilisateur
- **Design moderne** : Interface React avec Tailwind CSS
- **Responsive** : Adaptation parfaite à tous les appareils
- **Animations 3D** : Modèles 3D interactifs avec Three.js
- **Reconnaissance vocale** : Saisie vocale pour une expérience immersive
- **Streaming en temps réel** : Mise à jour dynamique des statuts

### 🚀 Performance
- **Optimisation GPU** : Support RTX 3050 avec CUDA
- **Streaming Server-Sent Events** : Communication bidirectionnelle en temps réel
- **Cache intelligent** : Gestion optimisée de la mémoire
- **Scalabilité** : Architecture modulaire et extensible

## 🏗️ Architecture

### Backend (Python/FastAPI)
```
backend/
├── app.py                 # API FastAPI principale
├── services/             # Services métier
│   ├── LLMService.py     # Service d'IA et LLM
│   ├── SerpService.py    # Service de recherche web
│   └── StreamGenerator.py # Générateur de flux temps réel
├── classifiers/          # Classificateurs d'intentions
├── Models/              # Modèles de données
└── config/              # Configuration système
```

### Frontend (React/TypeScript)
```
UI/
├── src/
│   ├── components/       # Composants réutilisables
│   ├── pages/           # Pages de l'application
│   ├── assets/          # Ressources statiques
│   └── style/           # Styles et CSS
├── public/              # Fichiers publics
└── package.json         # Dépendances Node.js
```

## 🚀 Installation

### Prérequis

- **Python 3.8+**
- **Node.js 18+**
- **CUDA Toolkit** (pour l'optimisation GPU)
- **Git**

### 1. Cloner le Repository

```bash
git clone https://github.com/votre-username/chatbot-fso.git
cd chatbot-fso
```

### 2. Configuration Backend

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Activer l'environnement (Linux/Mac)
source venv/bin/activate

# Installer les dépendances
cd backend
pip install -r requirements.txt

# Configuration GPU (optionnel)
# Assurez-vous que CUDA est installé et configuré
```

### 3. Configuration Frontend

```bash
# Installer les dépendances Node.js
cd UI
npm install

# Variables d'environnement
cp .env.example .env
# Modifier .env avec vos configurations
```

### 4. Lancement

#### Backend
```bash
cd backend
python start_server.py
# ou
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd UI
npm run dev
```

L'application sera accessible sur `http://localhost:5173`

## 💻 Utilisation

### 🎯 Interface Principale

1. **Accueil** : Page d'accueil avec présentation de la FSO
2. **Chat** : Interface de conversation avec l'assistant
3. **Navigation** : Menu responsive avec accès aux différentes sections

### 💬 Conversation avec le Chatbot

1. **Saisie** : Tapez votre question ou utilisez la reconnaissance vocale
2. **Traitement** : Le système analyse et classe votre demande
3. **Recherche** : Consultation de la base de données FSO
4. **Réponse** : Réponse structurée et contextuelle
5. **Fallback** : Si nécessaire, recherche web automatique

### 🎨 Fonctionnalités Avancées

- **Reconnaissance vocale** : Cliquez sur le microphone pour parler
- **Modèles 3D** : Visualisez des objets 3D interactifs
- **Streaming temps réel** : Suivez le traitement de votre demande
- **Interface multilingue** : Basculez entre français et anglais

## 🔧 Configuration

### Variables d'Environnement

```bash
# Backend (.env)
CUDA_VISIBLE_DEVICES=1
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:latest

# Frontend (.env)
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=Chatbot FSO
```

### Configuration GPU

Le système est optimisé pour les cartes graphiques NVIDIA RTX :

```python
# Configuration automatique dans LLMService.py
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['OLLAMA_GPU_LAYERS'] = '999'
os.environ['NVIDIA_VISIBLE_DEVICES'] = '1'
```

### Modèles IA

Le système utilise Ollama avec le modèle `gpt-oss:latest`. Assurez-vous qu'Ollama est installé et en cours d'exécution :

```bash
# Installation Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Téléchargement du modèle
ollama pull gpt-oss:latest

# Démarrage du service
ollama serve
```

## 📁 Structure du Projet

```
CHATBOT-FSO/
├── 📁 backend/                    # Backend Python/FastAPI
│   ├── 📁 classifiers/           # Classificateurs ML
│   ├── 📁 config/                # Configuration système
│   ├── 📁 data/                  # Données et datasets
│   ├── 📁 Models/                # Modèles de données
│   ├── 📁 services/              # Services métier
│   ├── app.py                    # Application principale
│   ├── requirements.txt          # Dépendances Python
│   └── start_server.py           # Script de démarrage
├── 📁 UI/                        # Frontend React/TypeScript
│   ├── 📁 src/                   # Code source
│   │   ├── 📁 components/        # Composants React
│   │   ├── 📁 pages/             # Pages de l'application
│   │   ├── 📁 assets/            # Ressources statiques
│   │   └── 📁 style/             # Styles CSS
│   ├── 📁 public/                # Fichiers publics
│   ├── package.json              # Dépendances Node.js
│   └── vite.config.ts            # Configuration Vite
└── 📄 README.md                  # Ce fichier
```

## 🤝 Contribution

Nous accueillons chaleureusement les contributions ! Voici comment participer :

### 🐛 Signaler un Bug

1. Vérifiez que le bug n'a pas déjà été signalé
2. Créez une issue avec une description détaillée
3. Incluez les étapes de reproduction et captures d'écran

### 💡 Proposer une Amélioration

1. Créez une issue pour discuter de votre idée
2. Attendez la validation de l'équipe
3. Implémentez votre solution

### 🔧 Développement

1. Fork le projet
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Créez une Pull Request

### 📋 Standards de Code

- **Python** : PEP 8, docstrings, type hints
- **TypeScript** : ESLint, Prettier, interfaces explicites
- **Git** : Messages de commit conventionnels
- **Tests** : Couverture de code > 80%

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- **Faculté des Sciences d'Oujda** pour le support et les données
- **Communauté open source** pour les outils et bibliothèques
- **Équipe de développement** pour le travail et l'innovation

## 📞 Contact

- **Email** : [rhouibi.ibti.fst@uhp.ac.ma](mailto:rhouibiibtissam@gmail.com)
              [bdaarkangelm@gmail.com](mailto:bdaarkangelm@gmail.com)  
---

<div align="center">

**🌟 N'oubliez pas de donner une étoile au projet si vous l'aimez ! 🌟**

*Développé avec ❤️ pour la Faculté des Sciences d'Oujda*

</div>
