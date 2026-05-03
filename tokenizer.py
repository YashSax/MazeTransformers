import enum
from typing import List

class Tokens(enum.Enum):
    # Path tokens
    TOKEN_RIGHT = 0
    TOKEN_LEFT  = 1
    TOKEN_UP    = 2
    TOKEN_DOWN  = 3
    TOKEN_EOS   = 4

    # Maze tokens
    TOKEN_WALL        = 5
    TOKEN_PATH        = 6
    TOKEN_NEWLINE     = 7
    TOKEN_END_OF_MAZE = 8

    TOKEN_PAD = 9


TOKEN_MAP = {
    "down" : Tokens.TOKEN_DOWN,
    "up" : Tokens.TOKEN_UP,
    "right" : Tokens.TOKEN_RIGHT,
    "left" : Tokens.TOKEN_LEFT
}

def tokenize(maze: List[List[int]], path: List[str]):
    tokenized = []
    for row in maze:
        for cell in row:
            tokenized.append([Tokens.TOKEN_WALL.value, Tokens.TOKEN_PATH.value][cell])
        tokenized.append(Tokens.TOKEN_NEWLINE.value)

    tokenized.append(Tokens.TOKEN_END_OF_MAZE.value)
    maze_tokens = len(tokenized)
    tokenized.extend(TOKEN_MAP[direction].value for direction in path)
    tokenized.append(Tokens.TOKEN_EOS.value)
    return tokenized, maze_tokens


if __name__ == "__main__":
    from maze import generate_valid_maze

    maze, directions = generate_valid_maze(size=32)

    tokenized_maze = tokenize(maze, directions)
    print(tokenized_maze)
