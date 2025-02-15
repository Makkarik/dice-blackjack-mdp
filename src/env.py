"""Module for DiceBlackJack environment and associated Dealer class."""

import logging
import os
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

DEFAULT_STATE = np.array([0, 0, 0, True])

FIRST_DIE, SECOND_DIE, DICE_SUM = 0, 1, 2
BUST_VALUE = 21

ACTIONS = {
    "hit_first": 0,
    "hit_second": 1,
    "hit_sum": 2,
    "stack_first": 3,
    "stack_second": 4,
    "stack_sum": 5,
}

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
        # Why score dimension is 24?
        # Actually, there is a way to get to 27 by rolling 6 - 6, having 15 points for
        # the previous rolls. Same goes to other cases, when it is possible to end game
        # with victory by choosing the minimal die, but player decides to stack the sum.
        # Adding to this example, that scores 1, 2 and 3 are impossible (as 1 - 1 at the
        # first roll yields 4), we got 28 - 3 = 24.
        self.observation_space = gym.spaces.MultiDiscrete([24, 7, 7, 2])
        self.dealer = Dealer(self._roll_dice, dealer_th)
        self.render_mode = render_mode
        self.metadata["render_fps"] = fps
        self.done = True

        self._cwd = os.path.dirname(__file__)
        logger.info("Dice Blackjack environment has been initialized.")

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[tuple, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        self.dealer.reset()

        self.player_state = DEFAULT_STATE.copy()
        self.player_history = []
        self._player_min_score = 0

        self.dealer_state = DEFAULT_STATE.copy()
        self._dealer_min_score = 0
        self.dealer_history = [np.append(self.dealer_state.copy(), None)]

        self._chance = False
        self._blackjack = False
        self.done = False

        self.player_state[1:3] = self._roll_dice()
        # Check if the double value has been rolled
        if self.player_state[1] == self.player_state[2]:
            self._chance = True
        self.player_history.append(np.append(self.player_state.copy(), None))
        return self._get_observation(), {}

    def _roll_dice(self) -> np.ndarray:
        """Roll the dice.

        Returns
        -------
        dice : np.ndarray
            The array of dice values: [die_1, die_2]

        """
        return self.np_random.integers(low=1, high=6, size=2, endpoint=True)

    def step(self, action: int) -> tuple[tuple, float, bool, bool, dict[str, Any]]:
        """Make a step.

        Parameters
        ----------
        action : int
            An action to apply.

        Returns
        -------
        observation : tuple[tuple, float, bool, bool, dict[str, Any]]
            A Gymnasium-style return to the wrapper/training cycle.

        """
        if not self.action_space.contains(action):
            msg = f"Action '{action}' is not in action space!"
            raise ValueError(msg)

        if self.done:
            msg = "Environment must be reset before use!"
            raise Exception(msg)

        self._player_min_score = self._get_player_score(self.player_state)
        if self._player_min_score > BUST_VALUE:
            self.done = True

        else:
            is_double_score = self.player_state[3]
            # If the first roll, map actions to choosing the sum of dice
            action = 3 * (action // 3) + 2 if is_double_score else action
            die_idx = action % 3
            is_stack = action // 3

            # Record state for debug purpose
            previous_state = self.player_state.copy()

            if is_stack:
                # Add points to the score
                dice = self.player_state[1:3]
                dice = np.append(dice, dice.sum())
                self._chance = False
                if is_double_score:
                    self.player_state[0] = 2 * dice[die_idx]
                    self.player_state[3] = False
                else:
                    self.player_state[0] += dice[die_idx]
                self.player_state[1:3] = 0
                self._player_min_score = self._get_player_score(self.player_state)
                # Engage dealer
                self.dealer_history, self._dealer_min_score = self.dealer.play()
                self.dealer_state = self.dealer_history[-1][:-1]
                self.done = True
                # Get the observation
                logger.debug("Transition %s -> %s", previous_state, self.player_state)

            else:
                # Add points to the score
                dice = self.player_state[1:3]
                dice = np.append(dice, dice.sum())
                if is_double_score:
                    self.player_state[0] = 2 * dice[die_idx]
                    self.player_state[3] = False
                else:
                    self.player_state[0] += dice[die_idx]
                # Roll the dice!
                dice = self._roll_dice()
                # Check the blackjack
                if dice[0] == dice[1] and self._chance:
                    self._blackjack = True
                else:
                    self._chance = False
                    self._blackjack = False
                self.player_state[1:3] = dice
                self._player_min_score = self._get_player_score(self.player_state)
                logger.debug("Transition %s -> %s", previous_state, self.player_state)
            # Write action to the previous state
            self.player_history[-1][-1] = int(action)
            self.player_history.append(np.append(self.player_state.copy(), None))

        reward = self._calculate_reward()
        logger.debug(
            "State: %s, reward: %d, is done: %s",
            str(self.player_state),
            reward,
            self.done,
        )
        return self._get_observation(), reward, self.done, self.done, {}

    def _get_observation(self) -> tuple:
        """Get the state of the environment.

        Returns
        -------
        observation : tuple
            An observation of the environment.

        """
        if self.done:
            logger.info(
                "Player's rolls: %s",
                str(np.stack(self.player_history)).replace("\n", " ->"),
            )
            logger.info(
                "Dealer's rolls: %s",
                str(np.stack(self.dealer_history)).replace("\n", " ->"),
            )
        # Human-readable option enables default rendering
        if self.render_mode == "human":
            self.render()
        return tuple(map(int, self.player_state))

    def _get_player_score(self, state: np.ndarray) -> None:
        """Recalculate the player score."""
        if self._chance:
            min_score = 0
        elif state[0] > BUST_VALUE:
            min_score = state[0]
        elif state[3]:
            min_score = 2 * state[1:3].sum()
        else:
            min_score = state[0] + state[1:3].min()
        return min_score

    def _calculate_reward(self) -> float:
        """Reward function implementation."""
        if self._blackjack:
            self.done = True
            logger.info("Game ended with the Blackjack combination!")
            return 2.0  # Ultimate victory
        elif self._dealer_min_score > BUST_VALUE:
            logger.info(
                "Game ended with the dealer's bust of score %02d",
                self._dealer_min_score,
            )
            self.done = True
            return 1.0  # Dealer's bust
        elif self._player_min_score > BUST_VALUE:
            logger.info(
                "Game ended with the player's bust of score %02d",
                self._player_min_score,
            )
            self.done = True
            return -1.0  # Bust lose
        elif self.player_state[0] > self.dealer_state[0] and self.done:
            logger.info(
                "Game ended with the player's victory: %02d > %02d",
                self._player_min_score,
                self._dealer_min_score,
            )
            return 1.0  # Player's victory
        elif self.player_state[0] < self.dealer_state[0] and self.done:
            logger.info(
                "Game ended with the dealer's victory: %02d < %02d",
                self._player_min_score,
                self._dealer_min_score,
            )
            self.done = True
            return -1.0  # Score lose
        elif self.player_state[0] == self.dealer_state[0] and self.done:
            logger.info(
                "Game ended with the draw: %02d = %02d",
                self._player_min_score,
                self._dealer_min_score,
            )
            return 0.0  # Draw
        else:
            return 0.0  # Intermediate state

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
        if state[4] is not None and state[4] % 3 in {0, 2}:
            surface.blit(selection, (left + 6, top + 78))
        if state[4] is not None and state[4] % 3 in {1, 2}:
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
        if 2 * dice.sum() <= self._threshold:
            action = ACTIONS["stack_sum"]
        else:
            action = ACTIONS["hit_sum"]
        self._state_history.append(np.append(self._state, action))

        while self.total_score < self._threshold:
            self._state[0] = self.total_score
            action = None
            # Roll the dice
            dice = self._roll_generator()
            if self.total_score + dice.min() > BUST_VALUE:
                break

            elif self.total_score + dice.sum() <= BUST_VALUE:
                self.total_score += dice.sum()
                action = self._hit_or_stack() + ACTIONS["hit_sum"]  # Add 2 literally

            elif self.total_score + dice.max() <= BUST_VALUE:
                self.total_score += dice.max()
                action = self._hit_or_stack() + dice.argmax()

            elif self.total_score + dice.min() <= BUST_VALUE:
                self.total_score += dice.min()
                action = self._hit_or_stack() + dice.argmin()
            else:
                self.total_score += dice.min()
            self._state[1:3] = dice
            # If the roll is not first, change the value
            self._roll += 1
            self._state_history.append(np.append(self._state, action))
        self._state[3] = int(self._roll == 0)

        if self.total_score <= BUST_VALUE:
            self._state[0] = self.total_score
            self._state[1:3] = 0
            self._state_history.append(np.append(self._state, None))
        return self._state_history, self.total_score

    def _hit_or_stack(self) -> int:
        """Decide either hit or stack."""
        return 3 if self.total_score <= self._threshold else 0

    def reset(self):
        """Reset the dealer's internal state."""
        self._state = DEFAULT_STATE.copy()
        self._state_history = []
        self._roll = 0


# Test snippet for the environment
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
    state, _ = env.reset()
    done = False
    logger.info("Initial state: %s", state)
    if env.render_mode != "human":
        logger.debug("Output array: %s", env.render().shape)

    while not done:
        action = int(input(prompt))
        state, reward, done, _, _ = env.step(action)
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
