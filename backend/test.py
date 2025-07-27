import torch

print("📦 PyTorch version:", torch.__version__)
print("🧠 CUDA version:", torch.version.cuda)
print("🚀 CUDA available:", torch.cuda.is_available())
print("🔢 Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"🖥️  GPU {i} name:", torch.cuda.get_device_name(i))
else:
    print("❌ No CUDA-compatible GPU detected.")
