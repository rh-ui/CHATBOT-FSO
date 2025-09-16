import os, sys, asyncio, subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# import torch


# print("🔍 Vérification CUDA...")
# print(f"CUDA disponible: {torch.cuda.is_available()}")
# print(f"Nombre de GPUs: {torch.cuda.device_count()}")
# if torch.cuda.is_available():
#     print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    
    
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    subprocess.run(cmd, check=True)
