import argparse
import json
import os
import shutil

from tqdm import tqdm

from maze import convert_to_directions, generate_maze, solve_maze


def generate_mazes(size: int, num_mazes: int, output_dir: str):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for i in tqdm(range(num_mazes)):
        path = None
        while path is None:
            maze = generate_maze(size=size)
            path = solve_maze(maze)

        with open(os.path.join(output_dir, f"maze_{i}.json"), "w") as f:
            json.dump({"maze": maze, "path": convert_to_directions(path)}, f)


def generate_dataset(size: int, num_train: int, num_test: int, output_dir: str):
    generate_mazes(size, num_train, os.path.join(output_dir, "train"))
    generate_mazes(size, num_test, os.path.join(output_dir, "test"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--num_mazes", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    generate_dataset(args.size, args.num_mazes, args.output_dir)
