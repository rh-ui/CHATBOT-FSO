import os
import time
import subprocess

# Configuration pour RTX 3050
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['NVIDIA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Importer votre service
from LLMService import llm_service

def test_gpu_usage():
    print("=== Test GPU RTX 3050 ===")
    
    # Afficher les GPUs disponibles
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    print("GPUs disponibles:")
    print(result.stdout)
    
    # Test de génération
    start = time.time()
    response = llm_service.generate_structured_response(
        question="Qu'est-ce que la physique quantique ?",
        search_results=[{"answer": "La physique quantique étudie les phénomènes à l'échelle atomique.", "score": 0.8}],
        lang='fr'
    )
    end = time.time()
    
    print(f"\nTemps de génération: {end - start:.2f}s")
    print(f"Réponse générée: {len(response['response'])} caractères")
    
    # Vérifier l'utilisation GPU
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    print("\nUtilisation GPU:")
    print(result.stdout)

if __name__ == "__main__":
    test_gpu_usage()