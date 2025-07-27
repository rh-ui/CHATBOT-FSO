import os
import sys

# Définir les variables d'environnement AVANT tout import
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Test rapide CUDA
import torch
print("🔍 Vérification CUDA...")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Nombre de GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")

# Import de l'application
import uvicorn
import app

if __name__ == "__main__":
    print("\n🚀 Démarrage du serveur avec uvicorn.run()...")
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=["./"]  # Pour le hot reload
    )