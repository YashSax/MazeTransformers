import argparse
import torch
from torch import Tensor
from typing import Dict
from model import MazeTransformer
import yaml
from supervised import MazeDataset, maze_collate_fn
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from generate_dataset import generate_dataset
from training_utils import create_wandb_run, remove_solution_from_sequences


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


def run_grpo(model, model_name, config, data_dir, wandb_run=None, update_every=1):
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
        cumulative_train_reward = cumulative_test_reward = 0
        num_train_batches = num_test_batches = 0

        model.train()
        for batch in tqdm(train_dataloader):
            sequences, _, sizes, dataset_names = batch
            input_sequences = remove_trai

            print(sequences.shape) # Should be (B, seq_len)
            assert False


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-wl", "--wandb_log", action="store_true")
    args.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    args.add_argument("-nd", "--no_data_generation", action="store_true")

    args = args.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    TRAINING_MODE = "grpo"

    if not args.no_data_generation:
        generate_dataset(config["dataset"][TRAINING_MODE])

    model_config = config["model"]
    training_config = config["training"]
    if not os.path.exists(training_config[TRAINING_MODE]["output_dir"]):
        os.makedirs(training_config[TRAINING_MODE]["output_dir"])

    model_output_dir = os.path.join(
        training_config[TRAINING_MODE]["output_dir"], model_config["name"]
    )
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        os.makedirs(os.path.join(model_output_dir, "checkpoints"))

    with open(os.path.join(model_output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    model = MazeTransformer(config["model"]).to(config["model"]["device"])
    torch.compile(model)

    run = create_wandb_run(config) if args.wandb_log else None
    run_grpo(
        model,
        model_config["name"],
        config["training"][TRAINING_MODE],
        config["dataset"][TRAINING_MODE]["output_dir"],
        run,
    )
