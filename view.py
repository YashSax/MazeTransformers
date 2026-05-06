import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 0 = wall, 1 = open, 2 = path, 3 = start, 4 = end
maze = np.ones((32, 32), dtype=int)

# Example walls (replace with your real maze)
maze[5, 0:20] = 0
maze[10:20, 15] = 0

# Example path (list of (row, col) tuples from your solver)
path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2)]
for r, c in path:
    maze[r, c] = 2

# Start and end (set these last so they overwrite the path color)
maze[0, 0] = 3
maze[31, 31] = 4

cmap = ListedColormap(["black", "white", "deepskyblue", "limegreen", "crimson"])

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(maze, cmap=cmap, vmin=0, vmax=4)

# Crisp grid lines between cells
ax.set_xticks(np.arange(-0.5, 32, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 32, 1), minor=True)
ax.grid(which="minor", color="gray", linewidth=0.5)
ax.tick_params(
    which="both", bottom=False, left=False, labelbottom=False, labelleft=False
)

plt.savefig("maze.png", dpi=150, bbox_inches="tight")
plt.show()
