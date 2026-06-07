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
from tokenizer import Tokens
from training_utils import (create_causal_mask, create_wandb_run,
                            remove_solution_from_sequences)


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


def calculate_rewards(
    predicted_paths: List[Tensor], actual_paths: List[Tensor], device: str | None = None
):
    rewards = torch.tensor(
        [
            calculate_reward(pred, actual)
            for pred, actual in zip(predicted_paths, actual_paths)
        ],
        dtype=torch.float32,
    )
    return rewards if device is None else rewards.to(device)


def calculate_advantages(
    predicted_paths: List[Tensor],
    actual_paths: List[Tensor],
    group_size: int,
    batch_size: int,
    device: str | None = None,
):
    rewards = calculate_rewards(predicted_paths, actual_paths, device=device)
    base = torch.arange(group_size, device=rewards.device).repeat(batch_size, 1) * batch_size
    offset = torch.arange(batch_size, device=rewards.device).reshape(batch_size, 1)
    group_row_idxs = base + offset

    group_mean_rewards = rewards[group_row_idxs].mean(dim=1).repeat(group_size)
    group_mean_std = rewards[group_row_idxs].std(dim=1).repeat(group_size) + 1e-8
    advantages = (rewards - group_mean_rewards) / group_mean_std
    return advantages, rewards


def build_rollout_batch(
    input_sequences: Tensor, sizes: IntTensor, predictions: List[Tensor]
) -> tuple[Tensor, Tensor]:
    max_rollout_len = max((prediction.shape[0] for prediction in predictions), default=0)
    batch_size, prompt_seq_len = input_sequences.shape
    rollout_sequences = torch.full(
        (batch_size, prompt_seq_len + max_rollout_len),
        Tokens.TOKEN_PAD.value,
        dtype=input_sequences.dtype,
        device=input_sequences.device,
    )
    rollout_sequences[:, :prompt_seq_len] = input_sequences
    rollout_lengths = torch.zeros(batch_size, dtype=torch.long, device=input_sequences.device)

    for idx, prediction in enumerate(predictions):
        rollout_len = prediction.shape[0]
        rollout_lengths[idx] = rollout_len
        if rollout_len == 0:
            continue

        start = int(sizes[idx].item())
        rollout_sequences[idx, start : start + rollout_len] = prediction.to(
            input_sequences.device
        )

    return rollout_sequences, rollout_lengths


def calculate_rollout_log_probs(
    model: MazeTransformer,
    rollout_sequences: Tensor,
    sizes: IntTensor,
    rollout_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    max_rollout_len = int(rollout_lengths.max().item()) if rollout_lengths.numel() else 0
    if max_rollout_len == 0:
        empty = torch.zeros((rollout_sequences.shape[0], 0), device=rollout_sequences.device)
        return empty, empty.bool()

    causal_masks = create_causal_mask(sizes, rollout_sequences.shape[1]).to(model.device)
    logits = model.forward(rollout_sequences, causal_masks)
    token_log_probs = torch.log_softmax(logits, dim=-1)

    gathered_log_probs = torch.zeros(
        (rollout_sequences.shape[0], max_rollout_len), device=model.device
    )
    rollout_mask = torch.zeros(
        (rollout_sequences.shape[0], max_rollout_len),
        dtype=torch.bool,
        device=model.device,
    )
    for idx in range(rollout_sequences.shape[0]):
        rollout_len = int(rollout_lengths[idx].item())
        if rollout_len == 0:
            continue

        start = int(sizes[idx].item())
        action_positions = torch.arange(rollout_len, device=model.device) + start
        logits_positions = action_positions - 1
        sampled_actions = rollout_sequences[idx, action_positions]
        gathered_log_probs[idx, :rollout_len] = token_log_probs[
            idx, logits_positions, sampled_actions
        ]
        rollout_mask[idx, :rollout_len] = True

    return gathered_log_probs, rollout_mask


def stack_rollout_log_probs(
    rollout_action_probs: List[Tensor], max_rollout_len: int, device: str
) -> Tensor:
    stacked_log_probs = torch.zeros(
        (len(rollout_action_probs), max_rollout_len), dtype=torch.float32, device=device
    )
    for idx, action_probs in enumerate(rollout_action_probs):
        if action_probs.numel() == 0:
            continue

        rollout_len = action_probs.shape[0]
        stacked_log_probs[idx, :rollout_len] = torch.log(
            action_probs.to(device).clamp_min(1e-8)
        )

    return stacked_log_probs


def calculate_completion_rate(predicted_paths: List[Tensor], actual_paths: List[Tensor]) -> float:
    completions = [
        predicted.shape == actual.shape and torch.equal(predicted, actual)
        for predicted, actual in zip(predicted_paths, actual_paths)
    ]
    return sum(completions) / len(completions)


def run_grpo(
    model: MazeTransformer, model_name, config, data_dir, wandb_run=None, update_every=5
):
    # For epoch in epochs
    # If epoch % update_baseline_every == 0: update the baseline model
    # For each batch:
    # For each item in the batch, sample group_size rollouts
    # For each of the rollouts, calculate the baseline probs
    # Calculate KL divergence penalty + importance sampling ratio
    # Calculate advantages
    # Calculate GRPO loss -> loss.backward()

    # Technically ^ not entirely accurate because we're doing multiple backwards passes for every rollout
    # TODO (Yash): Write a better description

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
    best_test_reward = float("-inf")
    best_state_dict = deepcopy(model.state_dict())

    for epoch in range(config["num_epochs"]):
        cumulative_train_reward = cumulative_test_reward = 0
        cumulative_train_loss = 0
        train_completion_rate = test_completion_rate = 0
        num_train_batches = num_test_batches = 0

        for batch in tqdm(train_dataloader):
            sequences, targets, sizes, dataset_names = batch
            batch_size = sizes.shape[0]
            input_sequences = remove_solution_from_sequences(sequences.clone(), sizes)
            input_sequences = input_sequences.to(config["device"])
            duplicated_input_sequences = input_sequences.repeat(config["group_size"], 1)
            duplicated_sizes = sizes.repeat(config["group_size"])
            duplicated_targets = list(targets) * config["group_size"]

            model.eval()
            predictions, pred_token_probs, ref_token_probs = model.generate_rollouts(
                duplicated_input_sequences,
                duplicated_sizes,
                temperature=config["sampling_temperature"],
                baseline_model=baseline_model,
            )
            rollout_sequences, rollout_lengths = build_rollout_batch(
                duplicated_input_sequences, duplicated_sizes, predictions
            )
            max_rollout_len = int(rollout_lengths.max().item()) if rollout_lengths.numel() else 0
            old_token_log_probs = stack_rollout_log_probs(
                pred_token_probs, max_rollout_len, config["device"]
            )
            ref_token_log_probs = stack_rollout_log_probs(
                ref_token_probs, max_rollout_len, config["device"]
            )

            advantages, rewards = calculate_advantages(
                predictions,
                duplicated_targets,
                config["group_size"],
                batch_size,
                device=config["device"],
            )

            advantages = advantages.unsqueeze(1)
            rollout_mask = (
                torch.arange(max_rollout_len, device=config["device"]).unsqueeze(0)
                < rollout_lengths.unsqueeze(1)
            ).float()
            normalizer = rollout_mask.sum().clamp_min(1.0)

            batch_loss = 0.0
            for _ in range(update_every):
                model.train()
                current_token_log_probs, _ = calculate_rollout_log_probs(
                    model, rollout_sequences, duplicated_sizes, rollout_lengths
                )

                importance_ratio = torch.exp(current_token_log_probs - old_token_log_probs)
                clipped_ratio = torch.clamp(
                    importance_ratio, 1 - config["eps"], 1 + config["eps"]
                )
                surrogate_objective = torch.minimum(
                    importance_ratio * advantages, clipped_ratio * advantages
                )

                log_ratio = current_token_log_probs - ref_token_log_probs
                kl_penalty = torch.exp(log_ratio) - log_ratio - 1

                policy_loss = -(surrogate_objective * rollout_mask).sum() / normalizer
                kl_loss = (kl_penalty * rollout_mask).sum() / normalizer
                loss = policy_loss + config["kl_penalty"] * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            cumulative_train_reward += rewards.mean().item()
            cumulative_train_loss += batch_loss / update_every
            train_completion_rate += calculate_completion_rate(
                predictions, duplicated_targets
            )
            num_train_batches += 1

        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                sequences, targets, sizes, dataset_names = batch
                input_sequences = remove_solution_from_sequences(sequences.clone(), sizes)
                input_sequences = input_sequences.to(config["device"])
                predictions, _ = model.generate_rollouts(
                    input_sequences,
                    sizes,
                    method="greedy",
                )
                rewards = calculate_rewards(
                    predictions, list(targets), device=config["device"]
                )
                cumulative_test_reward += rewards.mean().item()
                test_completion_rate += calculate_completion_rate(
                    predictions, list(targets)
                )
                num_test_batches += 1

        avg_train_loss = cumulative_train_loss / num_train_batches
        avg_train_reward = cumulative_train_reward / num_train_batches
        avg_test_reward = cumulative_test_reward / num_test_batches
        avg_train_completion = train_completion_rate / num_train_batches
        avg_test_completion = test_completion_rate / num_test_batches

        if avg_test_reward > best_test_reward:
            best_test_reward = avg_test_reward
            best_state_dict = deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1}: average train loss = {avg_train_loss}")
        print(f"Epoch {epoch + 1}: average train reward = {avg_train_reward}")
        print(f"Epoch {epoch + 1}: average test reward = {avg_test_reward}")
        print(f"Epoch {epoch + 1}: train completion rate = {avg_train_completion}")
        print(f"Epoch {epoch + 1}: test completion rate = {avg_test_completion}")
        print("-" * 80)

        if wandb_run:
            wandb_run.log(
                {
                    "avg_train_loss": avg_train_loss,
                    "avg_train_reward": avg_train_reward,
                    "avg_test_reward": avg_test_reward,
                    "avg_train_completion": avg_train_completion,
                    "avg_test_completion": avg_test_completion,
                }
            )

        if (epoch + 1) % config["baseline_update_frequency"] == 0:
            baseline_model.load_state_dict(model.state_dict())
            baseline_model.eval()

    torch.save(best_state_dict, os.path.join(config["output_dir"], model_name, "model"))


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
