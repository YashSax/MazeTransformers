import torch
from torch import Tensor
from tokenizer import Tokens
from typing import Dict
import wandb

def create_causal_mask(maze_sizes: Tensor, seq_len: int):
    masks = []
    for size in maze_sizes:
        mask = torch.ones((seq_len, seq_len)).tril().bool()
        mask[:size, :size] = True
        masks.append(mask)
    return torch.stack(masks).unsqueeze(1)

def remove_solution_from_sequences(sequences: Tensor, sizes: Tensor) -> Tensor:
    _, seq_len = sequences.shape
    cols = torch.arange(seq_len)
    mask = cols[None, :] >= sizes[:, None]

    sequences[mask] = Tokens.TOKEN_PAD.value
    sequences = sequences[:, :sizes.max()]
    return sequences

def create_wandb_run(config: Dict):
    return wandb.init(
        entity=config["entity"],
        project=config["project"],
        name=f"run: {config['name']}, model: {config['model']['name']}",
        config=config,
    )
