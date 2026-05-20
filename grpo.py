import argparse
import torch
from torch import Tensor
from typing import Dict
from model import MazeTransformer
import yaml
from supervised import MazeDataset
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy


def reward(predicted_path: Tensor, actual_path: Tensor):
    # Let's start off with the simplest reward, which is just +1 if
    # the predicted direction matches the correct direction.
    # Eventually we can do things like offer partial rewards for
    # going into an empty space (instead of into a wall).
    # We then have an additional reward for ending at the right time.

    # Maybe there's a cute prefix/cumsum way to do this but for now we're
    # just going to do the inefficient thing and manually iterate through
    # the tensor and find the first disagreement.

    min_path_length = min(predicted_path.shape[0], actual_path.shape[0])
    for idx in range(min_path_length):
        if predicted_path[idx] != actual_path[idx]:
            return idx
    return min_path_length + (predicted_path.shape[0] == actual_path.shape[0])


def grpo(model, model_name, config, data_dir, wandb_run=None, save_every=5):
    # For epoch in epochs
    # If epoch % update_baseline_every == 0: update the baseline model
    # For each batch:
    # For each item in the batch, sample group_size rollouts
    # For each of the rollouts, calculate the baseline probs
    # Calculate KL divergence penalty + importance sampling ratio
    # Calculate advantages
    # Calculate GRPO loss -> loss.backward()

    train_dataset = MazeDataset(os.path.join(data_dir, "train"))
    test_dataset = MazeDataset(os.path.join(data_dir, "test"))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=maze_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=maze_collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    best_test_loss = 1e99
    best_state_dict = None
    baseline_model = deepcopy(model)
    for epoch in range(config["num_epochs"]):
        cumulative_train_loss = cumulative_test_loss = 0
        num_train_batches = num_test_batches = 0
        train_completions = defaultdict(list)
        test_completions = defaultdict(list)

        if epoch % config["baseline_update_frequency"] == 0:
            baseline_model = model.copy()

        model.train()
        for batch in tqdm(train_dataloader):
            pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-wl", "--wandb_log", action="store_true")
    args.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    args.add_argument("-nd", "--no_data_generation", action="store_true")

    args = args.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = MazeTransformer(config["model"]).to(config["model"]["device"])
    grpo(model, None, config, None)
