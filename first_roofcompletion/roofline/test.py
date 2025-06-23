import torch
path = "experiments/finetune_roof_completion_250416_002016/checkpoint/52_Network.pth"
model = torch.load(path)
print(model.keys())