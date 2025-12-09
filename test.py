import torch 
ckpt = torch.load("/home/y_yu/LlamaGen/ckpt/c2i_B_256.pt", map_location="cpu") 
print(ckpt.keys())