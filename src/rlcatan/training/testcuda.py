import torch

print("CUDA version PyTorch sees:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
