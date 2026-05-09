import argparse
from collections import defaultdict
import json
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import wandb
from generate_dataset import generate_dataset
from model import MazeTransformer
from tokenizer import Tokens, tokenize

from tqdm import tqdm

def create_wandb_run(config: Dict):
    return wandb.init(
        entity=config["entity"],
        project=config["project"],
        name=f"run: {config["name"]}, model: {config["model"]["name"]}",
        config=config,
    )


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
        dataset_name = self.data[idx]["dataset_name"]

        tokenized, maze_token_size = tokenize(maze, path)
        tokenized = torch.tensor(tokenized)
        maze_token_size = torch.tensor(maze_token_size)
        target = tokenized[maze_token_size:]
        return tokenized, target, maze_token_size, dataset_name


def maze_collate_fn(batch):
    # All tensors in the batch need to be the same size -> pad to largest
    sequences, targets, maze_token_sizes, dataset_names = list(zip(*batch))
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=Tokens.TOKEN_PAD.value
    )
    padded_maze_token_sizes = torch.tensor(maze_token_sizes)
    return padded_sequences, targets, padded_maze_token_sizes, dataset_names


def create_causal_mask(maze_sizes: Tensor, seq_len: int):
    masks = []
    for size in maze_sizes:
        mask = torch.ones((seq_len, seq_len)).tril().bool()
        mask[:size, :size] = True
        masks.append(mask)
    return torch.stack(masks).unsqueeze(1)


def calculate_loss(
    model_output: Tensor,
    targets: Tuple[Tensor],
    maze_sizes: Tensor,
    dataset_names: Tuple[str],
    device="mps"
):
    # This is super inefficient, there should be a way for targets to be a padded Tensor
    # Loss function shouldn't be putting things on the device.
    dataset_completion_cumulative = defaultdict(list)
    loss = 0
    num_elements = 0
    for idx, target in enumerate(targets):
        target = target.to(device)
        num_elements += target.numel()
        full_prediction = model_output[idx]
        maze_size = maze_sizes[idx]
        dataset_name = dataset_names[idx]

        start = maze_size - 1
        predicted = full_prediction[start : start + target.shape[0]]
        dataset_completion_cumulative[dataset_name].append((torch.argmax(predicted, dim=1) == target).all())

        loss += F.cross_entropy(predicted, target, reduction="none").sum()

    loss /= num_elements

    return loss, dataset_completion_cumulative


def train(config, data_dir, wandb_run=None, save_every=5):
    train_dataset = MazeDataset(os.path.join(data_dir, "train"))
    test_dataset = MazeDataset(os.path.join(data_dir, "test"))
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=maze_collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=maze_collate_fn
    )
    model = MazeTransformer(config).to(config["device"])
    torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    best_test_loss = 1e99
    best_state_dict = None

    for epoch in range(config["num_epochs"]):
        cumulative_train_loss = cumulative_test_loss = 0
        num_train_batches = num_test_batches = 0
        train_completions = defaultdict(list)
        test_completions = defaultdict(list)

        model.train()
        for batch in tqdm(train_dataloader):
            sequences, targets, sizes, dataset_names = batch
            seq_len = sequences.shape[1]
            masks = create_causal_mask(sizes, seq_len)
            sequences = sequences.to(config["device"])
            masks = masks.to(config["device"])

            optimizer.zero_grad()
            model_out = model.forward(sequences, masks)
            loss, batch_completion_cumulative = calculate_loss(
                model_out, targets, sizes, dataset_names, device=config["device"]
            )

            for dataset_name, completions in batch_completion_cumulative.items():
                train_completions[dataset_name].extend(completions)

            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item()
            num_train_batches += 1

        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                sequences, targets, sizes, dataset_names = batch
                seq_len = sequences.shape[1]
                masks = create_causal_mask(sizes, seq_len)

                sequences = sequences.to(config["device"])
                masks = masks.to(config["device"])

                model_out = model.forward(sequences, masks)
                loss, batch_completion_cumulative = calculate_loss(
                    model_out, targets, sizes, dataset_names, device=config["device"]
                )

                for dataset_name, completions in batch_completion_cumulative.items():
                    test_completions[dataset_name].extend(completions)

                cumulative_test_loss += loss.item()
                num_test_batches += 1

        avg_train_loss = cumulative_train_loss / num_train_batches
        avg_test_loss = cumulative_test_loss / num_test_batches

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_state_dict = model.state_dict()

        train_completion_rates = {f"{name}_train_completion" : sum(completed) / len(completed) for name, completed in train_completions.items()}
        test_completion_rates = {f"{name}_test_completion" : sum(completed) / len(completed) for name, completed in test_completions.items()}

        print(f"Epoch {epoch + 1}: average train loss = {avg_train_loss}")
        print(f"Epoch {epoch + 1}: average test loss = {avg_test_loss}")
        for ds_name, completion_rate in train_completion_rates.items():
            print(f"Epoch {epoch + 1}: {ds_name} completion rate = {completion_rate}")

        for ds_name, completion_rate in test_completion_rates.items():
            print(f"Epoch {epoch + 1}: {ds_name} completion rate = {completion_rate}")

        print("-" * 80)

        if wandb_run:
            run.log(
                {
                    "avg_train_loss": avg_train_loss,
                    "avg_test_loss": avg_test_loss,
                    **train_completion_rates,
                    **test_completion_rates
                }
            )

        if epoch + 1 % save_every == 0:
            torch.save(
                best_state_dict,
                os.path.join(
                    config["output_dir"],
                    config["name"],
                    "checkpoints",
                    f"checkpoint_{epoch + 1}",
                ),
            )

    torch.save(
        best_state_dict, os.path.join(config["output_dir"], config["name"], "model")
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-wl", "--wandb_log", action="store_true")
    args.add_argument("-c", "--config", type=str, default="configs/config.yaml")

    args = args.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # generate_dataset(config["dataset"])

    model_config = config["model"]
    if not os.path.exists(model_config["output_dir"]):
        os.makedirs(model_config["output_dir"])

    if not os.path.exists(os.path.join(model_config["output_dir"], model_config["name"])):
        os.makedirs(os.path.join(model_config["output_dir"], model_config["name"]))
        os.makedirs(os.path.join(model_config["output_dir"], model_config["name"], "checkpoints"))

    with open(
        os.path.join(model_config["output_dir"], model_config["name"], "config.yaml"), "w"
    ) as f:
        yaml.safe_dump(config, f)

    run = create_wandb_run(config) if args.wandb_log else None
    train(config["model"], config["dataset"]["output_dir"], run)
