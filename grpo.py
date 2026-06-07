import argparse
import os
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

import torch
import yaml
from torch import Tensor, IntTensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from generate_dataset import generate_dataset
from model import MazeTransformer
from supervised import MazeDataset, maze_collate_fn
from training_utils import create_wandb_run, remove_solution_from_sequences


def calculate_reward(predicted_path: Tensor, actual_path: Tensor):
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


def calculate_advantages(
    predicted_paths: List[Tensor],
    actual_paths: List[Tensor],
    group_size: int,
    batch_size: int,
):
    base = torch.arange(group_size).repeat(batch_size, 1) * batch_size
    offset = torch.arange(batch_size).reshape(batch_size, 1)
    group_row_idxs = base + offset

    rewards = torch.Tensor(
        [
            calculate_reward(pred, actual)
            for pred, actual in zip(predicted_paths, actual_paths)
        ]
    )

    group_mean_rewards = rewards[group_row_idxs].mean(dim=1).repeat(group_size)
    group_mean_std = rewards[group_row_idxs].std(dim=1).repeat(group_size) + 1e-8
    advantages = (rewards - group_mean_rewards) / group_mean_std
    return advantages


def run_grpo(
    model: MazeTransformer, model_name, config, data_dir, wandb_run=None, update_every=1
):
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

    baseline_model = deepcopy(model)
    baseline_model.eval()

    for epoch in range(config["num_epochs"]):
        cumulative_train_reward = cumulative_test_reward = 0
        num_train_batches = num_test_batches = 0

        model.train()
        for batch in tqdm(train_dataloader):
            sequences, targets, sizes, dataset_names = batch
            input_sequences = remove_solution_from_sequences(sequences, sizes)
            input_sequences = input_sequences.to(config["device"])
            duplicated_input_sequences = input_sequences.repeat(config["group_size"], 1)
            duplicated_sizes = sizes.repeat(config["group_size"])
            duplicated_targets = list(targets) * config["group_size"]

            predictions, pred_token_probs, ref_token_probs = model.generate_rollouts(
                duplicated_input_sequences,
                duplicated_sizes,
                temperature=config["sampling_temperature"],
                baseline_model=baseline_model
            )

            advantages = calculate_advantages(
                predictions,
                duplicated_targets,
                config["group_size"],
                config["batch_size"],
            )

            pred_token_logprobs = [torch.log(probs) for probs in pred_token_probs]
            ref_token_logprobs = [torch.log(probs) for probs in ref_token_probs]
            kl_divergence = [pred - ref for pred, ref in zip(pred_token_logprobs, ref_token_logprobs)]


            print(advantages)
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
