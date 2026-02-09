# check if torch is using gpu
import torch
print(torch.cuda.is_available())
# check how many gpus are available
print(torch.cuda.device_count())

