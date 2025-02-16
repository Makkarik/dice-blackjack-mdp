"""Visualization utilities for Q-learning agent."""

import os

import gymnasium as gym
import imageio
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure


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


def show_training_stats(
    env: gym.Wrapper, loss: list[float], roll_length: int = 1000
) -> Figure:
    """Show the training loop statistics.

    Parameters
    ----------
    env : gym.Wrapper
        An environment wrapper.
    loss : list[float]
        A list of the step-wise losses.
    roll_length : int = 1000
        A length of rolling average.

    Returns
    -------
    fig : Figure
        A figure to save.

    """
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
    axs = axs.flatten()
    axs[0].set_title("Episode rewards", fontsize=14)
    axs[0].grid()
    axs[0].plot(*moving_average(env.return_queue, roll_length))
    axs[0].axhline(0, color="red", linestyle="--", label="Tie frontier")
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

    return fig


def show_inference_stats(env: gym.Wrapper, roll_length: int = 1000) -> Figure:
    """Show the inference statistics.

    Parameters
    ----------
    env : gym.Wrapper
        An environment wrapper.
    loss : list[float]
        A list of the step-wise losses.
    roll_length : int = 1000
        A length of rolling average.

    Returns
    -------
    fig : Figure
        A figure to save.

    """
    fig, axs = plt.subplots(ncols=2, figsize=(16, 4))
    axs = axs.flatten()
    axs[0].set_title("Episode rewards", fontsize=14)
    axs[0].grid()
    axs[0].plot(*moving_average(env.return_queue, roll_length))
    axs[0].axhline(0, color="red", linestyle="--", label="Tie frontier")
    axs[0].axhline(
        np.mean(env.return_queue), color="C1", linestyle="--", label="Mean reward"
    )
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

    plt.tight_layout()
    plt.show()

    return fig


def convert_q_table(q_table: dict[str, np.ndarray]) -> dict[tuple[int, np.ndarray]]:
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
    # Restore state space, using the values from the Q-table
    state_space = np.stack(list(map(np.array, q_table.keys()))).max(axis=0)
    # Create the tensor with the dimensions of the state space
    action_tensor = -1 * np.ones(shape=state_space + 1)
    # Translate Q-table values to the action tensor
    for key, values in q_table.items():
        if np.all(values == 0):
            action_tensor[*tuple(key)] = -1  # No policy found for that state
        else:
            action_tensor[*tuple(key)] = np.argmax(values)
    # Transform action tensor to the action table, as for some scores there is no
    # possible state at all
    action_table = {}
    for score in range(state_space[0]):
        grid = action_tensor[score, 1:, 1:]  # Drop zero values
        if not np.all(grid < 0):
            action_table[score] = grid

    return action_table


def plot_policy_tables_18(action_table, action_info) -> Figure:
    """Build heatmaps based on the dictionary tables_data.

    Parameters
    ----------
    action_table : dict
        Dictionary where the key is 'previous score' (a total of 18 keys),
        and the value is a 6 x 6 matrix with the action values.
    action_info : dict
        Dictionary where the key is the action code and the value is a tuple
        containing the action label and the corresponding color.

    Returns
    -------
    fig : Figure
        A figure to save.


    """
    keys = sorted(action_table.keys())
    n_rows, n_cols = 3, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12))

    cmap = ListedColormap(["gray"] + [action_info[i][1] for i in range(6)])
    norm = BoundaryNorm(np.arange(-1.5, 6.5, 1), cmap.N)

    for ax, key in zip(axes.flat, keys):
        actions = action_table[key]
        annots = np.vectorize(lambda x: action_info[x][0] if x > -1 else "")(actions)
        sns.heatmap(
            actions,
            fmt="",
            cmap=cmap,
            norm=norm,
            cbar=False,
            xticklabels=np.arange(1, 7),
            yticklabels=np.arange(1, 7),
            ax=ax,
            annot=annots,
            linewidth=0.1,
            linecolor="black",
            annot_kws={"fontsize": 12, "color": "black"},
        )

        ax.set_title(f"Prev. Score = {key}", fontsize=14)
        ax.set_xlabel("Second die")
        ax.set_ylabel("First die")
        ax.invert_yaxis()

    fig.suptitle("Policy for Dice Blackjack", fontsize=18)
    plt.tight_layout()
    plt.show()

    return fig


def mp4_to_gif(folder: str) -> None:
    """Convert MP4 video to GIF."""
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")]
    gif_paths = [p[: p.rfind(".")] + ".gif" for p in paths]

    for video_path, gif_path in zip(paths, gif_paths):
        with imageio.get_reader(video_path) as reader:
            fps = reader.get_meta_data()["fps"]

            writer = imageio.get_writer(gif_path, fps=fps)
            for frame in reader:
                writer.append_data(frame)
            writer.close()

        os.remove(video_path)
