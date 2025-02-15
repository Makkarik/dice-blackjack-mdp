"""The implementation of the different RL agents."""

from collections import defaultdict
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from tqdm import trange


class QLearningAgent:
    """RL agent for solving the Dice Blackjack.

    Attributes
    ----------
    q_values : dict
        A table of Q-values.
    lr : float
        A learning rate.
    discount_factor : float
        A discount factor for computing the Q-values.
    epsilon : float
        The initial stochasticity value.
    epsilon_decay : float
        The decay for the epsilon.
    final_epsilon : float
        The final stochasticity value.
    random_generator : np.random.Generator
        The internal random generator.
    training_error : list[int]
        A list of the training errors during the runs.

    Methods
    -------
    get_action(env: gym.Env, obs: np.ndarray) -> int
        A method, that returns the action for the given observation.
    update(obs: np.ndarray, action: int, reward: float, terminated: bool,
           ext_obs: np.ndarray) -> None
        Updates the corresponding Q-value.
    episode_step() ->
        Updates the espsilon rate (must be called after each run).

    """

    def __init__(
        self,
        env,
        n_episodes: int,
        seed: int | None = None,
    ):
        """Initialize an RL agent, utilizing the Q-Learning.

        Parameters
        ----------
        env : gymnasium.Env
            The agent's environmet.
        n_episodes : int
            The number of episodes for training.
        seed : int | None = None
            The seed for internal random generator.

        """
        self.env = env
        self.n_episodes = n_episodes
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        """Reset agent's internal state."""
        self.random_generator = np.random.default_rng(seed=self.seed)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.training_error = []
        self.trained = False

    def get_action(self, env: gym.Env, obs: np.ndarray) -> int:
        """Return the best action with probability (1 - epsilon) or a random one.

        Parameters
        ----------
        env : gymnasium.Env
            An environment to send the action to
        obs : np.ndarray
            An observation, obtained from the environment.

        Returns
        -------
        action : int
            An output action.

        """
        # With probability epsilon return a random action to explore the environment
        # (only if the agent has not been trained before)
        if not self.trained and self.random_generator.random() < self._epsilon:
            action = env.action_space.sample()
        # With probability (1 - epsilon) act greedily (exploit)
        else:
            # a = ARGMAX(Q, s)
            action = int(np.argmax(self.q_table[obs]))
        return action

    def _update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        """Update the Q-value of an action.

        Parameters
        ----------
        obs : tuple
            The previous observation.
        action : int
            The action that has been applied to environment.
        reward : float
            The reward from the applied action.
        terminated : bool
            The flag for episode termination.
        next_obs : tuple
            The next observation.

        """
        # Calculate MAX Q[s', :] if the episode is not terminated
        q_max = (not terminated) * np.max(self.q_table[next_obs])
        # Compute dQ := R[s, a] + gamma * MAX Q[s', :] - Q[s, a]
        delta = reward + self.gamma * q_max - self.q_table[obs][action]
        # Update Q[s, a] := Q[s, a] + lr * dQ
        self.q_table[obs][action] += self.lr * delta
        self.training_error.append(delta)

    def train(
        self,
        learning_rate: float,
        discount_factor: float = 0.95,
        epsilon_decay: Callable[[float], float] | None = None,
        progress: bool = True,
    ) -> list[float]:
        """Training loop for the agent.

        Parameters
        ----------
        learning_rate : float
            The learning rate for the training.
        discount_factor : float = 0.95
            The discount factor for computing the Q-value
        epsilon_decay: Callable | None = None
            A function, that map training progress [0, 1) to the epsilon value [0, 1].
        progress : bool = True
            Progressbar toggle.

        Returns
        -------
        training_error : list[float]
            List of the training errors.

        """
        self.lr = learning_rate
        self.gamma = discount_factor

        if epsilon_decay is None:
            epsilon_decay = lambda x: max(1 - x, 1e-3)

        for episode in trange(self.n_episodes, desc="Training", disable=not progress):
            # Initialize the environment with the pseudorandom value
            obs, _ = self.env.reset(
                seed=int(self.random_generator.integers(0, 2**32 - 1))
            )
            self._epsilon = epsilon_decay(episode / self.n_episodes)
            done = False

            # Play one episode
            while not done:
                action = self.get_action(self.env, tuple(obs))
                next_obs, reward, done, _, _ = self.env.step(action)
                # Update the Q-value
                self._update(obs, action, reward, done, next_obs)
                obs = next_obs

        # Make the agent deterministic at inference
        self.trained = True
        return self.training_error

    def get_q_table(self) -> dict[tuple]:
        """Return a copy of table with the Q-values.

        Returns
        -------
        q_table : dict
            The table of Q-values

        """
        if not self.trained:
            msg = "Agent must be trained first!"
            raise Exception(msg)

        return self.q_table.copy()
