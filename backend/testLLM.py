# import requests
# import json
# import time

# # Configuration
# API_URL = "http://localhost:8000"

# # Questions de test
# questions_test = [
#     # "ما هو برنامج الماجستير المتخصص في علوم الأغذية-السلامة الصحية؟",
#     # "Quel est le module final du Master CHAP Opt MPA en 2ème année ?",
#     "What is the Faculty of Science at Mohammed Premier University?"
#     # "Comment créer un compte ?",
#     # "Mot de passe oublié",
#     # "Comment contacter le support ?",
#     # "SMI"
#     # "Quelles sont les heures d'ouverture ?",
#     # "How to reset password?",
#     # "كيف أنشئ حساب جديد؟",
#     # "Prix des services",
#     # "Question qui n'existe pas dans la base"
# ]

# def test_question(question, use_llm=True):
#     """Teste une question avec l'API"""
#     print(f"\n{'='*50}")
#     print(f"Question: {question}")
#     print(f"LLM activé: {use_llm}")
#     print(f"{'='*50}")
    
#     payload = {
#         "question": question,
#         "use_llm": use_llm,
#         "k": 3,
#         "score_threshold": 0.3
#     }
    
#     try:
#         start_time = time.time()
#         response = requests.post(f"{API_URL}/search", json=payload)
#         end_time = time.time()
        
#         if response.status_code == 200:
#             data = response.json()
            
#             print(f"⏱️  Temps de réponse: {end_time - start_time:.2f}s")
#             print(f"🌐 Langue détectée: {data.get('detected_lang', 'N/A')}")
            
#             if use_llm:
#                 print(f"📊 Résultats trouvés: {data.get('total_results_found', 0)}")
#                 print(f"🎯 Confiance: {data.get('confidence', 0):.2f}")
#                 print(f"📚 Sources utilisées: {data.get('sources_used', 0)}")
#                 print(f"\n💬 Réponse structurée:")
#                 print(f"   {data.get('structured_response', 'Pas de réponse')}")
                
#                 if data.get('raw_results'):
#                     print(f"\n📋 Résultats bruts:")
#                     for i, result in enumerate(data['raw_results'][:3], 1):
#                         print(f"   {i}. Q: {result['question']}")
#                         print(f"      R: {result['answer'][:100]}...")
#                         print(f"      Score: {result['score']:.2f}")
#             else:
#                 print(f"📊 Résultats trouvés: {data.get('total_found', 0)}")
#                 if data.get('results'):
#                     print(f"\n📋 Résultats:")
#                     for i, result in enumerate(data['results'][:3], 1):
#                         print(f"   {i}. Q: {result['question']}")
#                         print(f"      R: {result['answer'][:100]}...")
#                         print(f"      Score: {result['score']:.2f}")
                        
#         else:
#             print(f"❌ Erreur {response.status_code}: {response.text}")
            
#     except requests.exceptions.ConnectionError:
#         print("❌ Erreur: Impossible de se connecter à l'API")
#         print("   Vérifiez que l'API est lancée sur http://localhost:8000")
#     except Exception as e:
#         print(f"❌ Erreur: {str(e)}")

# def test_health():
#     """Teste le health check"""
#     print(f"\n{'='*50}")
#     print("TEST HEALTH CHECK")
#     print(f"{'='*50}")
    
#     try:
#         response = requests.get(f"{API_URL}/health")
#         if response.status_code == 200:
#             data = response.json()
#             print("-- API accessible --")
#             print(f"   OpenSearch: {'oui' if data.get('opensearch') else 'non'}")
#             print(f"   Modèle chargé: {'oui' if data.get('model_loaded') else 'non'}")
#             print(f"   Service LLM: {'oui' if data.get('llm_service') else 'non'}")
#         else:
#             print(f"❌ Health check échoué: {response.status_code}")
#     except Exception as e:
#         print(f"❌ Erreur health check: {str(e)}")

# def test_chat_endpoint():
#     """Teste l'endpoint chat"""
#     print(f"\n{'='*50}")
#     print("TEST ENDPOINT CHAT")
#     print(f"{'='*50}")
    
#     payload = {
#         "question": "Comment puis-je vous aider ?",
#         "context": {
#             "user_type": "nouveau",
#             "session_id": "test"
#         }
#     }
    
#     try:
#         response = requests.post(f"{API_URL}/chat", json=payload)
#         if response.status_code == 200:
#             data = response.json()
#             print("✅ Endpoint chat fonctionnel")
#             print(f"   LLM utilisé: {data.get('llm_used')}")
#             print(f"   Réponse: {data.get('structured_response', 'N/A')[:100]}...")
#         else:
#             print(f"❌ Erreur chat: {response.status_code}")
#     except Exception as e:
#         print(f"❌ Erreur chat: {str(e)}")

# def main():
#     """Programme principal"""
#     print("🚀 TEST SIMPLE DE L'API FAQ CHATBOT")
#     print("=" * 60)
    
#     # Test de santé
#     test_health()
    
#     # Test endpoint chat
#     test_chat_endpoint()
    
#     # Test des questions avec LLM
#     print(f"\n{'🤖 TESTS AVEC LLM':^60}")
#     for question in questions_test:
#         test_question(question, use_llm=True)
#         time.sleep(0.5)
    
#     # Test quelques questions sans LLM
#     # print(f"\n{'🔍 TESTS SANS LLM':^60}")
#     # for question in questions_test[:3]:
#     #     test_question(question, use_llm=False)
#     #     time.sleep(0.5)
    
#     print(f"\n{'✅ TESTS TERMINÉS':^60}")

# if __name__ == "__main__":
#     main()





# """
# Script de test pour vérifier la connexion à Ollama et le modèle deepseek-r1
# """

# import requests
# import json
# import time

# def test_ollama_connection():
#     """Test la connexion à Ollama"""
#     try:
#         print("🔍 Test de connexion à Ollama...")
#         response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
#         if response.status_code == 200:
#             print("✅ Connexion à Ollama réussie")
#             models = response.json()
#             print(f"📋 Modèles disponibles : {len(models.get('models', []))}")
            
#             for model in models.get('models', []):
#                 print(f"   - {model['name']}")
            
#             return True
#         else:
#             print(f"❌ Erreur HTTP {response.status_code}")
#             return False
            
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Impossible de se connecter à Ollama: {str(e)}")
#         print("💡 Assurez-vous qu'Ollama est en cours d'exécution avec: ollama serve")
#         return False

# def test_deepseek_model():
#     """Test le modèle deepseek-r1"""
#     try:
#         print("\n🤖 Test du modèle deepseek-r1...")
        
#         payload = {
#             "model": "deepseek-r1",
#             "prompt": "c'est quoi la faculté des sciences oujda?",
#             "stream": False,
#             "options": {
#                 "temperature": 0.1,
#                 "num_predict": 100
#             }
#         }
        
#         print("⏳ Envoi de la requête...")
#         start_time = time.time()
        
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json=payload,
#             timeout=60
#         )
        
#         processing_time = time.time() - start_time
        
#         if response.status_code == 200:
#             result = response.json()
#             print(f"✅ Réponse reçue en {processing_time:.2f}s")
#             print(f"🗣️ Réponse: {result.get('response', 'Pas de réponse')}")
#             return True
#         else:
#             print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
#             return False
            
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Erreur lors du test du modèle: {str(e)}")
#         return False

# def pull_deepseek_model():
#     """Télécharge le modèle deepseek-r1 si nécessaire"""
#     try:
#         print("\n📥 Vérification/téléchargement du modèle deepseek-r1...")
        
#         payload = {"name": "deepseek-r1"}
        
#         response = requests.post(
#             "http://localhost:11434/api/pull",
#             json=payload,
#             timeout=300
#         )
        
#         if response.status_code == 200:
#             print("✅ Modèle deepseek-r1 disponible")
#             return True
#         else:
#             print(f"❌ Erreur lors du téléchargement: {response.status_code}")
#             return False
            
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Erreur lors du téléchargement: {str(e)}")
#         return False

# def main():
#     """Fonction principale de test"""
#     print("🚀 Test de configuration Ollama + DeepSeek-R1")
#     print("=" * 50)
    
#     # Test de connexion
#     if not test_ollama_connection():
#         print("\n💡 Pour démarrer Ollama:")
#         print("   ollama serve")
#         return
    
#     # Test du modèle
#     if not test_deepseek_model():
#         print("\n💡 Pour télécharger le modèle:")
#         print("   ollama pull deepseek-r1")
        
#         # Tentative de téléchargement automatique
#         if pull_deepseek_model():
#             test_deepseek_model()
    
#     print("\n🎉 Configuration terminée!")
#     print("Vous pouvez maintenant utiliser votre chatbot avec Ollama + DeepSeek-R1")

# if __name__ == "__main__":
#     main()


import requests
import json

API_URL = "http://localhost:8000/search"

test_questions = [
    "Qui est le doyen de la faculté des Sciences de l'Université Mohammed Premier?"
]

def test_question(question, lang="fr", use_llm=True):
    print(f"\n🎯 Question: {question}")
    try:
        response = requests.post(API_URL, json={
            "question": question,
            "lang": lang,
            "use_llm": use_llm
        })

        if response.status_code != 200:
            print(f"❌ Erreur HTTP: {response.status_code} - {response.text}")
            return

        data = response.json()

        print("✅ Langue détectée:", data.get("detected_lang", "?"))
        print("🧠 LLM utilisé:", data.get("llm_used"))
        print("📊 Score confiance:", data.get("confidence"))
        print("📚 Sources utilisées:", data.get("sources_used"))
        print("🕒 Temps:", data.get("processing_time"), "s")
        print("\n💬 Réponse:\n", data.get("structured_response", "Aucune"))

    except Exception as e:
        print("❌ Exception:", str(e))


if __name__ == "__main__":
    print("🚀 TEST DU CHATBOT BACKEND AVEC OLLAMA\n")

    for q in test_questions:
        test_question(q)

    print("\n✅ Test terminé.")
