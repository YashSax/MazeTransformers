from maze import generate_maze
from tqdm import tqdm

MAZE_SIZE = 8

seen = set()
for i in tqdm(range(100_000)):
    maze = generate_maze(MAZE_SIZE)

    combined = []
    for row in maze:
        combined.extend(row)

    combined = tuple(combined)
    seen.add(combined)

print(len(seen))