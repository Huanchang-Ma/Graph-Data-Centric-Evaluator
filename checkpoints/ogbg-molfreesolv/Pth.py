import torch


file_path = "./Jun11-14:03:38_400.pth"


checkpoint = torch.load(file_path)


print(f"File: {file_path}")
print(checkpoint)
