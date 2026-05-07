import argparse
import json
import os
import shutil
from typing import Dict

from torch.utils.data import dataset
from tqdm import tqdm

from maze import convert_to_directions, generate_maze, solve_maze


def generate_mazes(dataset_name: str, size: int, num_mazes: int, output_dir: str):
    for i in tqdm(range(num_mazes)):
        path = None
        while path is None:
            maze = generate_maze(size=size)
            path = solve_maze(maze)

        with open(os.path.join(output_dir, f"maze_{i}.json"), "w") as f:
            json.dump({
                "maze": maze,
                "path": convert_to_directions(path),
                "dataset_name" : dataset_name
            }, f)


def generate_dataset(dataset_config: Dict):
    assert "output_dir" in dataset_config, "the dataset config must have an \"output_dir\"!"
    shutil.rmtree(dataset_config["output_dir"], ignore_errors=True)
    os.makedirs(dataset_config["output_dir"])

    splits = ["train", "test"]
    for split in splits:
        os.makedirs(os.path.join(dataset_config["output_dir"], split))

    for dataset_name, dataset_info in dataset_config.items():
        if type(dataset_info) != dict: # ignore the `output_dir`
            continue

        print("Generating data for", dataset_name)
        for split in splits:
            generate_mazes(
                dataset_name=dataset_name,
                size=dataset_info["size"],
                num_mazes=dataset_info[f"num_{split}"],
                output_dir=os.path.join(dataset_config["output_dir"], split)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--num_mazes", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    generate_dataset(args.size, args.num_mazes, args.output_dir)
