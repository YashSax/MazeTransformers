import argparse
import json
import os
from typing import Any, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from generate_dataset import generate_dataset
from model import MazeTransformer
from tokenizer import Tokens, tokenize


class MazeDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

        self.data = []
        for file in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, file), "r") as f:
                data = json.load(f)
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        maze = self.data[idx]["maze"]
        path = self.data[idx]["path"]

        tokenized, maze_token_size = tokenize(maze, path)
        tokenized = torch.tensor(tokenized)
        maze_token_size = torch.tensor(maze_token_size)
        target = tokenized[maze_token_size:]
        return tokenized, target, maze_token_size


def maze_collate_fn(batch):
    sequences, targets, maze_token_sizes = list(zip(*batch))
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=Tokens.TOKEN_PAD.value
    )
    padded_maze_token_sizes = torch.tensor(maze_token_sizes)
    return padded_sequences, targets, padded_maze_token_sizes


def create_causal_mask(maze_sizes: Tensor, seq_len: int):
    masks = []
    for size in maze_sizes:
        mask = torch.ones((seq_len, seq_len)).tril()
        mask[:size, :size] = True
        masks.append(mask)
    return torch.stack(masks).unsqueeze(1)


def calculate_loss(
    model_output: Tensor, targets: Tensor, maze_sizes: Tuple[Tensor], device="mps"
):  # TODO: check the math on this -> if I were to guess why learning isn't slow it's prob this
    # This is super inefficient, there should be a way for targets to be a padded Tensor
    # Loss function shouldn't be putting things on the device.
    loss = 0
    num_elements = 0
    for idx, target in enumerate(targets):
        target = target.to(device)
        num_elements += target.numel()
        full_prediction = model_output[idx]
        maze_size = maze_sizes[idx]

        start = maze_size - 1
        predicted = full_prediction[start : start + target.shape[0]]
        loss += F.cross_entropy(predicted, target, reduction="none").sum()

    loss /= num_elements
    return loss


def train(config, data_dir):
    train_dataset = MazeDataset(os.path.join(data_dir, "train"))
    test_dataset = MazeDataset(os.path.join(data_dir, "test"))
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=maze_collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, collate_fn=maze_collate_fn
    )
    model = MazeTransformer(config).to(config["device"])
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        cumulative_train_loss = cumulative_test_loss = 0
        num_train_batches = num_test_batches = 0

        model.train()
        for batch in train_dataloader:
            sequences, targets, sizes = batch
            seq_len = sequences.shape[1]
            masks = create_causal_mask(sizes, seq_len)

            sequences = sequences.to(config["device"])
            masks = masks.to(config["device"])

            optimizer.zero_grad()
            model_out = model.forward(sequences, masks)
            loss = calculate_loss(model_out, targets, sizes, device=config["device"])

            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item()
            num_train_batches += 1

        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                sequences, targets, sizes = batch
                seq_len = sequences.shape[1]
                masks = create_causal_mask(sizes, seq_len)

                sequences = sequences.to(config["device"])
                masks = masks.to(config["device"])

                model_out = model.forward(sequences, masks)
                loss = calculate_loss(
                    model_out, targets, sizes, device=config["device"]
                )

                cumulative_test_loss += loss.item()
                num_test_batches += 1

        print(
            f"Epoch {epoch + 1}: average train loss = {cumulative_train_loss / num_train_batches}"
        )
        print(
            f"Epoch {epoch + 1}: average test loss = {cumulative_test_loss / num_test_batches}"
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="configs/config.yaml")

    args = args.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generate_dataset(**config["dataset"])

    train(config["model"], config["dataset"]["output_dir"])
