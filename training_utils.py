import torch
from torch import Tensor

def create_causal_mask(maze_sizes: Tensor, seq_len: int):
    masks = []
    for size in maze_sizes:
        mask = torch.ones((seq_len, seq_len)).tril().bool()
        mask[:size, :size] = True
        masks.append(mask)
    return torch.stack(masks).unsqueeze(1)
