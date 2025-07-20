import torch
print(torch.cuda.is_available())           # Should print: True
print(torch.cuda.device_count())           # Should print: 2
print(torch.cuda.get_device_name(0))       # Should print: NVIDIA GeForce RTX 4090
print(torch.cuda.get_device_name(1))       # Should print: NVIDIA GeForce RTX 4090
