from multiprocessing import parent_process
import random
from typing import Tuple, List, Dict, Optional
import enum
from collections import deque
from termcolor import colored

DEFAULT_SIZE = 32


class Direction(enum.Enum):
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP = (-1, 0)
    DOWN = (1, 0)


def generate_maze(size: int = DEFAULT_SIZE) -> List[List[int]]:
    maze = [[0 for _ in range(size)] for _ in range(size)]
    maze[0][0] = 1

    current_square = (0, 0)

    dfs(maze, current_square)

    return maze


def get_surrounding_squares(
    maze: List[List[int]], square: Tuple[int, int]
) -> List[Tuple[int, int]]:
    surrounding_squares = []
    is_in_bounds = lambda square: 0 <= square[0] < len(maze) and 0 <= square[1] < len(
        maze[0]
    )
    if is_in_bounds((square[0] - 1, square[1])):
        surrounding_squares.append((square[0] - 1, square[1]))
    if is_in_bounds((square[0] + 1, square[1])):
        surrounding_squares.append((square[0] + 1, square[1]))
    if is_in_bounds((square[0], square[1] - 1)):
        surrounding_squares.append((square[0], square[1] - 1))
    if is_in_bounds((square[0], square[1] + 1)):
        surrounding_squares.append((square[0], square[1] + 1))
    return surrounding_squares


def can_create_in_direction(
    maze: List[List[int]], square: Tuple[int, int], direction: Direction
) -> bool:
    new_square = (square[0] + direction.value[0], square[1] + direction.value[1])
    if not (0 <= new_square[0] < len(maze) and 0 <= new_square[1] < len(maze[0])):
        return False

    if maze[new_square[0]][new_square[1]] != 0:
        return False

    num_filled_neighbors = sum(
        maze[i][j] == 1 for i, j in get_surrounding_squares(maze, new_square)
    )
    assert (
        num_filled_neighbors != 0
    ), "`can_move_in_direction` was called on a square with no filled neighbors!"
    return num_filled_neighbors <= 1


def can_move_in_direction(
    maze: List[List[int]], square: Tuple[int, int], direction: Direction
) -> List[Direction]:
    new_square = (square[0] + direction.value[0], square[1] + direction.value[1])
    return (
        0 <= new_square[0] < len(maze)
        and 0 <= new_square[1] < len(maze[0])
        and maze[new_square[0]][new_square[1]] == 1
    )


def dfs(
    maze: List[List[int]],
    current_square: Tuple[int, int],
) -> None:
    maze[current_square[0]][current_square[1]] = 1

    valid_directions = [
        direction
        for direction in Direction
        if can_create_in_direction(maze, current_square, direction)
    ]
    random.shuffle(valid_directions)
    for direction in valid_directions:
        if not can_create_in_direction(maze, current_square, direction):
            continue

        new_square = (
            current_square[0] + direction.value[0],
            current_square[1] + direction.value[1],
        )
        dfs(maze, new_square)


def print_maze(maze: List[List[int]], path: Optional[List[Tuple[int, int]]] = None):
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if path and (i, j) in path:
                print(colored("#", "green"), end=" ")
            else:
                print("#" if cell == 1 else ".", end=" ")
        print()


def reconstruct_path(
    parents: Dict[Tuple[int, int], Tuple[int, int] | None], end: Tuple[int, int] | None
) -> List[Tuple[int, int]]:
    if end == None:
        return []

    return reconstruct_path(parents, parents[end]) + [end]


def solve_maze(maze: List[List[int]]) -> Optional[List[Tuple[int, int]]]:
    start, end = (0, 0), (len(maze) - 1, len(maze[0]) - 1)
    queue = deque([start])
    visited = set([start])
    parents = {start: None}

    while queue:
        current_square = queue.popleft()
        if current_square == end:
            return reconstruct_path(parents, end)

        valid_directions = [
            direction
            for direction in Direction
            if can_move_in_direction(maze, current_square, direction)
        ]
        for direction in valid_directions:
            new_square = (
                current_square[0] + direction.value[0],
                current_square[1] + direction.value[1],
            )
            if new_square in visited:
                continue

            visited.add(new_square)
            queue.append(new_square)
            parents[new_square] = current_square

    return None


def convert_to_directions(path: List[Tuple[int, int]]) -> List[str]:
    directions = []
    for before, after in zip(path, path[1:]):
        if before[0] < after[0]:
            directions.append("down")
        elif before[0] > after[0]:
            directions.append("up")
        elif before[1] < after[1]:
            directions.append("right")
        elif before[1] > after[1]:
            directions.append("left")
        else:
            raise ValueError(f"Invalid path: {path}")
    return directions


def generate_valid_maze(size):
    path = None

    while not path:
        maze = generate_maze(size=size)
        path = solve_maze(maze)

    return maze, convert_to_directions(path)


if __name__ == "__main__":
    maze = generate_maze(size=32)

    path = solve_maze(maze)
    print_maze(maze, path)

    print(convert_to_directions(path))
