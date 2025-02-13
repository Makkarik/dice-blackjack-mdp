"""Module for DiceBlackJack environment and associated Dealer class."""

import logging
import os
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

DEFAULT_STATE = np.array([0, 0, 0, True], dtype=np.uint8)

FIRST_DIE, SECOND_DIE, DICE_SUM = 0, 1, 2
BUST_VALUE = 21

WHITE = (253, 253, 253)

logger = logging.getLogger(__name__)


class DiceBlackJack(gym.Env):
    """Dice Blackjack RL environment implementation."""

    def __init__(
        self, dealer_th: int = 17, render_mode: str | None = None, fps: int = 10
    ):
        """Initialize the environment."""
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.MultiDiscrete([25, 7, 7, 2])
        self.dealer = Dealer(self._roll_dice, dealer_th)
        self.render_mode = render_mode
        self.metadata["render_fps"] = fps
        self.done = True

        self._cwd = os.path.dirname(__file__)
        logger.info("Dice Blackjack environment has been initialized.")

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int, bool]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.dealer.reset()

        self.player_state = DEFAULT_STATE.copy()
        self.player_history = []
        self._player_min_score = 0

        self.dealer_state = DEFAULT_STATE.copy()
        self._dealer_min_score = 0
        self.dealer_history = [np.append(self.dealer_state, None)]

        self._chance_for_blackjack = False
        self._blackjack = False
        self.done = False

        self.player_state[1:3] = self._roll_dice()
        self._check_blackjack()
        self.player_history.append(np.append(self.player_state.copy(), DICE_SUM))
        return self._get_observation()

    def _roll_dice(self) -> np.ndarray:
        """Roll the dice.

        Returns
        -------
        dice : np.ndarray
            The array of dice values: [die_1, die_2]

        """
        return self.np_random.integers(low=1, high=6, size=2, endpoint=True)

    def step(self, action):
        """Make a step."""
        if not self.action_space.contains(action):
            msg = f"Action '{action}' is not in action space!"
            raise ValueError(msg)

        if self.done:
            msg = "Environment must be reset before use!"
            raise Exception(msg)

        is_double_score = self.player_state[3]
        die_idx = 2 if is_double_score else action % 3
        is_stack = action // 3

        # Record state for debug purpose
        previous_state = self.player_state.copy()

        if is_stack:
            # Add points to the score
            dice = self.player_state[1:3]
            dice = np.append(dice, dice.sum())
            if is_double_score:
                self.player_state[0] = 2 * dice[die_idx]
            else:
                self.player_state[0] += dice[die_idx]
            self.player_state[1:3] = 0
            # Engage dealer
            self.dealer_history, self._dealer_min_score = self.dealer.play()
            self.dealer_state = self.dealer_history[-1]
            self.done = True
            # Get the observation
            logger.debug("Transition %s -> %s", previous_state, self.player_state)

        else:
            # Add points to the score
            dice = self.player_state[1:3]
            dice = np.append(dice, dice.sum())
            if is_double_score:
                self.player_state[0] = 2 * dice[die_idx]
            else:
                self.player_state[0] += dice[die_idx]
            self.player_state[3] = False  # It is not the first roll
            # Roll the dice!
            dice = self._roll_dice()
            self.player_state[1:3] = dice
            # Get the observation
            self._check_blackjack()
            logger.debug("Transition %s -> %s", previous_state, self.player_state)

        self.player_history.append(np.append(self.player_state.copy(), None))
        if not self.player_history[-2][-2]:
            self.player_history[-2][-1] = action
        return self._get_observation()

    def _check_blackjack(self) -> None:
        """Check the Blackjack combination."""
        # If player has rolled double value in a row, he get the Blackjack
        if self.player_state[1] == self.player_state[2] and self._chance_for_blackjack:
            self._blackjack = True
        # If player has rolled double value once, he get the cahnce for Blackjack
        elif self.player_state[1] == self.player_state[2] and self.player_state[1] > 0:
            self._chance_for_blackjack = True
            self._blackjack = False
        # Any other case drop both variables
        else:
            self._chance_for_blackjack = False
            self._blackjack = False

    def _get_observation(self) -> None:
        """Get the state of the environment."""
        self._update_players_score()
        reward = self._calculate_reward()
        logger.debug(
            "State: %s, reward: %d, is done: %s",
            str(self.player_state),
            reward,
            self.done,
        )
        if self.done:
            logger.debug(
                "Player's rolls: %s",
                str(np.stack(self.player_history)).replace("\n", " ->"),
            )
            logger.debug(
                "Dealer's rolls: %s",
                str(np.stack(self.dealer_history)).replace("\n", " ->"),
            )
        # Human-readable option enables default rendering
        if self.render_mode == "human":
            self.render()
        return self.player_state, reward, self.done

    def _update_players_score(self) -> None:
        """Recalculate the player score if the bust is unavoidable."""
        min_score = 0
        if not self._chance_for_blackjack and not self.player_state[3]:
            min_score = self.player_state[0] + self.player_state[1:3].min()
        elif not self._chance_for_blackjack and self.player_state[3]:
            min_score = 2 * self.player_state[1:3].sum()

        if min_score > BUST_VALUE:
            self._player_min_score = min_score
            self.done = True
        else:
            self._player_min_score = self.player_state[0]

    def _calculate_reward(self) -> int:
        """Reward function implementation."""
        if self._blackjack:
            self.done = True
            logger.debug("Game ended with the Blackjack combination!")
            return 1  # Ultimate victory
        elif self._dealer_min_score > BUST_VALUE:
            logger.debug(
                "Game ended with the dealer's bust of score %02d",
                self._dealer_min_score,
            )
            self.done = True
            return 1  # Dealer's bust
        elif self._player_min_score > BUST_VALUE:
            logger.debug(
                "Game ended with the player's bust of score %02d",
                self._player_min_score,
            )
            self.done = True
            return -1  # Bust lose
        elif self._player_min_score > self._dealer_min_score and self.done:
            logger.debug(
                "Game ended with the player's victory: %02d > %02d",
                self._player_min_score,
                self._dealer_min_score,
            )
            return 1  # Player's victory
        elif self._player_min_score < self._dealer_min_score and self.done:
            logger.debug(
                "Game ended with the dealer's victory: %02d < %02d",
                self._player_min_score,
                self._dealer_min_score,
            )
            self.done = True
            return -1  # Score lose
        elif self._player_min_score == self._dealer_min_score and self.done:
            logger.debug(
                "Game ended with the draw: %02d = %02d",
                self._player_min_score,
                self._dealer_min_score,
            )
            return 0  # Draw
        else:
            return 0  # Intermediate state

    def render(self):
        """Render the environment.

        If `self.render_mode == "human"`, update the pygame display.
        Otherwise, return an RGB array representing the current frame.
        """
        if self.render_mode is None:
            # Warn if render mode is not set
            logger.warning(
                "You are calling render() without specifying a render mode. "
                "You can specify it at initialization, e.g. "
                'DiceBlackjack(render_mode="rgb_array")'
            )
            return

        # Window dimensions
        window_width, window_height = 920, 690

        # Initialize the display/screen if not already done
        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption("Dice Blackjack")
            else:
                pygame.font.init()
                self.screen = pygame.Surface((window_width, window_height))

        # Initialize the clock if not already done (for human mode fps control)
        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        # Place the image to background
        background = pygame.image.load(os.path.join(self._cwd, "assets/background.png"))
        self.screen.blit(background, (0, 0))
        # Get the font
        self._font = pygame.font.Font(
            os.path.join(self._cwd, "assets/Grand9K_Pixel.ttf"), 48
        )

        player_history = truncate_list(self.player_history)
        for i, roll in enumerate(player_history):
            self._draw_roll(self.screen, 139 + 120 * i, 360, roll)

        dealer_history = truncate_list(self.dealer_history)
        for i, roll in enumerate(dealer_history):
            self._draw_roll(self.screen, 139 + 120 * i, 36, roll)

        # --- Finalize the frame based on render mode ---
        if self.render_mode == "human":
            pygame.event.pump()  # Process event queue
            pygame.display.flip()
            # Use the throttled FPS
            self.clock.tick(self.metadata.get("render_fps"))
        else:
            # For non-human render mode, return an RGB array of the frame.
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _draw_roll(
        self, surface: pygame.Surface, left: int, top: int, state: np.ndarray
    ):
        """Draw a single roll on the given surface."""
        score = self._font.render(f"{state[0]:02d}", True, WHITE)
        score_rect = score.get_rect(center=(left + 60, top + 30))
        surface.blit(score, score_rect)

        selection = pygame.image.load(os.path.join(self._cwd, "assets/select.png"))
        if state[4] and state[4] % 3 in {0, 2}:
            surface.blit(selection, (left + 6, top + 78))
        if state[4] and state[4] % 3 in {1, 2}:
            surface.blit(selection, (left + 6, top + 78 + 108))

        for i, die in enumerate(state[1:3]):
            die = pygame.image.load(os.path.join(self._cwd, f"assets/die_{die}.png"))
            surface.blit(die, (left + 12, top + 84 + (108 * i)))


def truncate_list(history: list[Any], n: int = 6) -> list[Any]:
    """Truncate the list to last n elements."""
    if len(history) > n:
        return history[-n:]
    else:
        return history


class Dealer:
    """Dealer implementation for the Dice Blackjack environment.

    The class simulates the behaviour of the dealer.

    Attriibutes
    -----------
    _roll_generator : Callable
        An environment's method to roll the dice.
    _threshold : int = 17
        The threshold, until which dealer must roll the dice.
    _state : np.ndarray
        A state of the dealer.
    _done : bool
        If the dealer has already played or not.
    """

    def __init__(self, roll_generator: Callable, threshold: int = 17):
        """Init dealer."""
        self._roll_generator = roll_generator
        self._threshold = threshold
        self._state = DEFAULT_STATE.copy()
        self._state_history = []
        self.reset()

    def play(self) -> int:
        """Model the dealer's behavior.

        Returns
        -------
        dealer_state : np.ndarray
            The dealer's final state after all rolls.

        """
        if self._roll > 0:
            msg = "Dealer must be reset before next play."
            raise Exception(msg)

        dice = self._roll_generator()
        self._roll = 0
        self.total_score = 2 * dice.sum()  # First roll
        self._state[1:3] = dice
        self._state_history.append(np.append(self._state, DICE_SUM))

        while self.total_score < self._threshold:
            self._state[0] = self.total_score
            # Roll the dice
            dice = self._roll_generator()
            self.total_score += dice.sum()
            self._state[1:3] = dice
            # If the roll is not first, change the value
            self._roll += 1
            self._state_history.append(np.append(self._state, DICE_SUM))
        self._state[3] = int(self._roll == 0)
        if self.total_score <= BUST_VALUE:
            self._state[0] = self.total_score
            self._state[1:3] = 0
            self._state_history.append(np.append(self._state, None))
        return self._state_history, self.total_score

    def reset(self):
        """Reset the dealer's internal state."""
        self._state = DEFAULT_STATE.copy()
        self._roll = 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG, stream=sys.stdout, format="[%(levelname)s] %(message)s"
    )
    logging.logProcesses = False

    prompt = (
        "Available actions:\n0 - hit first die;   1 - hit second die;   2 - hit sum;\n"
        "3 - stack first die; 4 - stack second die; 5 - stack sum.\nEnter the action: "
    )
    env = DiceBlackJack(render_mode="human")
    state, reward, done = env.reset()
    logger.info("Initial state: %s, reward: %s, done: %s", state, reward, done)
    if env.render_mode != "human":
        logger.debug("Output array: %s", env.render().shape)

    while not done:
        action = int(input(prompt))
        state, reward, done = env.step(action)
        logger.info(
            "Action: %s, State: %s, Reward: %s, Done: %s", action, state, reward, done
        )
        if env.render_mode != "human":
            logger.debug("Output array: %s", env.render().shape)

    # Add a loop to keep the window open until the user closes it.
    if env.render_mode == "human":
        import keyboard

        logger.info("Game over. Press Enter to exit.")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if keyboard.is_pressed("enter"):
                running = False
    pygame.quit()
