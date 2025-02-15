"""Visualization utilities for Q-learning agent."""

import gymnasium as gym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


def moving_average(input: np.ndarray, n: int = 500, mode="valid") -> np.ndarray:
    """Get the moving average."""
    output = np.convolve(np.array(input).flatten(), np.ones(n), mode=mode) / n
    if mode == "valid":
        steps = np.arange(output.size) + n // 2
    elif mode == "same":
        steps = np.arange(output.size)
    return steps, output


def cumulative(input: np.ndarray) -> np.ndarray:
    """Get the cumulative value."""
    input = np.array(input).flatten()
    temp = 0
    for i in range(input.size):
        temp += input[i]
        input[i] = temp
    steps = np.arange(input.size)
    return steps, input


def show_training_stats(env: gym.Wrapper, loss: list[float], roll_length: int = 1000):
    """Show the training loop statistics.

    Parameters
    ----------
    env : gym.Wrapper
        An environment wrapper.
    loss : list[float]
        A list of the step-wise losses.
    roll_length : int = 1000
        A length of rolling average.

    """
    _, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
    axs = axs.flatten()
    axs[0].set_title("Episode rewards", fontsize=14)
    axs[0].grid()
    axs[0].plot(*moving_average(env.return_queue, roll_length))
    axs[0].axhline(0, color="red", linestyle="--", label="Breakeven frontier")
    axs[0].legend(fontsize=12)
    axs[0].set_xlabel("Episode", fontsize=12)
    axs[0].set_ylabel("Reward [points]", fontsize=12)

    axs[1].set_title("Cumulative episode rewards", fontsize=14)
    axs[1].grid()
    axs[1].plot(*cumulative(env.return_queue))
    axs[1].axhline(0, color="red", linestyle="--", label="Breakeven frontier")
    axs[1].legend(fontsize=12)
    axs[1].set_xlabel("Episode", fontsize=12)
    axs[1].set_ylabel("Cumulative reward [points]", fontsize=12)

    axs[2].set_title("Episode lengths", fontsize=14)
    axs[2].grid()
    axs[2].plot(*moving_average(env.length_queue, roll_length))
    axs[2].set_xlabel("Episode", fontsize=12)
    axs[2].set_ylabel("Length [steps]", fontsize=12)

    axs[3].set_title("Training Error", fontsize=14)
    axs[3].grid()
    axs[3].plot(*moving_average(loss, roll_length))
    axs[3].set_xlabel("Step", fontsize=12)
    axs[3].set_ylabel("Error value", fontsize=12)

    plt.tight_layout()
    plt.show()


def clean_q_table(q_table: dict[str, np.ndarray]) -> dict[tuple[int, np.ndarray]]:
    """Convert Q-table to the action table {str: int}.

    Parameters
    ----------
    q_table : dict[str, np.ndarray]
        The Q-table to be cleaned. Keys are state strings and values are numpy arrays
        of action values.

    Returns
    -------
    dict[tuple[int, np.ndarray]]
        A dictionary where keys are tuples of state components and values are lists
        of tuples containing state components and the index of the maximum action value.

    """
    lst = sorted(q_table.keys())
    tables_data = {}

    for i in lst:
        if not (q_table[i] == [0, 0, 0, 0, 0, 0]).all():
            if i[0] not in tables_data:
                tables_data[i[0]] = []
            tables_data[i[0]].append((i[1], i[2], np.argmax(q_table[i]).item()))
    return tables_data


def plot_tables_heatmaps_18(tables_dict, action_info):
    """Build heatmaps based on the dictionary tables_data.

    Parameters
    ----------
    tables_dict : dict
        Dictionary where the key is 'previous score' (a total of 18 keys),
        and the value is a list of tuples (d1, d2, action), where d1, d2 ∈ {1..6}
        and action ∈ {0..5}.
    action_info : dict
        Dictionary where the key is the action code and the value is a tuple
        containing the action label and the corresponding color.

    Returns
    -------
    None
        This function does not return anything. It displays the heatmaps.

    Notes
    -----
    For each key in `tables_dict`, it draws a separate 6x6 matrix, where:
      - The Y-axis (rows) represents d1 from 1 to 6,
      - The X-axis (columns) represents d2 from 1 to 6,
      - The cell color indicates the 'action code'.

    """
    keys = sorted(tables_dict.keys())
    n_keys = len(keys)

    n_rows, n_cols = 3, 6
    _, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12))

    cmap = ListedColormap(["gray"] + [action_info[i][1] for i in range(6)])
    norm = BoundaryNorm(np.arange(-1.5, 6.5, 1), cmap.N)

    for ax, key in zip(axes.flat, keys):
        grid = -np.ones((6, 6), dtype=int)
        annot = np.full((6, 6), "", dtype=object)

        for d1, d2, action_idx in tables_dict[key]:
            grid[d1 - 1, d2 - 1] = action_idx
            annot[d1 - 1, d2 - 1] = action_info[action_idx][0]
            ax.text(
                d2 - 0.5,
                d1 - 0.5,
                action_info[action_idx][0],
                ha="center",
                va="center",
                color="black",
                fontsize=11,
            )

        sns.heatmap(
            grid,
            fmt="",
            cmap=cmap,
            norm=norm,
            cbar=False,
            xticklabels=np.arange(1, 7),
            yticklabels=np.arange(1, 7),
            ax=ax,
        )

        ax.set_title(f"Prev. Score = {key}", fontsize=14)
        ax.set_xlabel("Second die")
        ax.set_ylabel("First die")
        ax.invert_yaxis()

    for i in range(n_keys, n_rows * n_cols):
        axes.flat[i].set_visible(False)

    plt.tight_layout()
    plt.show()
