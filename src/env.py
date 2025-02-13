"""Module for DiceBlackJack environment and associated Dealer class."""

import logging
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import pygame

DEFAULT_STATE = np.array([0, 0, 0, True], dtype=np.uint8)

FIRST_DIE, SECOND_DIE, DICE_SUM = 0, 1, 2
BUST_VALUE = 21

DICE_PIPS = {
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
    6: [
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.5),
        (0.75, 0.5),
        (0.25, 0.75),
        (0.75, 0.75),
    ],
}

GREEN = (50, 150, 50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

logger = logging.getLogger(__name__)


class DiceBlackJack(gym.Env):
    """Dice Blackjack RL environment implementation."""

    def __init__(self, dealer_th: int = 17):
        """Initialize the environment."""
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.MultiDiscrete([25, 7, 7, 2])
        self.dealer = Dealer(self._roll_dice, dealer_th)
        logger.info("Dice Blackjack environment has been initialized.")
        self.done = True

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

        self.player_history.append(np.append(self.player_state.copy(), action))
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
            logger.debug("Player's rolls: %s",
                         str(np.stack(self.player_history)).replace("\n", " ->"))
            logger.debug("Dealer's rolls: %s",
                         str(np.stack(self.dealer_history)).replace("\n", " ->"))
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
        """Render the Environment."""
        pygame.init()

        # Window size
        window_width, window_height = 600, 400
        screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Dice Blackjack")

        def draw_die(screen, center_x, center_y, value, size=100):
            """Draw the dice.

            Draws a single die at the given center (center_x, center_y)
            with a specified side size, labeled with 'value' (1..6).

            """
            # Die background
            left = center_x - size // 2
            top = center_y - size // 2
            pygame.draw.rect(screen, WHITE, (left, top, size, size), border_radius=10)
            pygame.draw.rect(
                screen, BLACK, (left, top, size, size), 2, border_radius=10
            )

            # Draw pips
            if value is None:
                # Draw a question mark in the center
                font = pygame.font.Font("./src/assets/Grand9K Pixel.ttf", size)
                text_surface = font.render("?", True, BLACK)
                tx = center_x - text_surface.get_width() // 2
                ty = center_y - text_surface.get_height() // 2
                screen.blit(text_surface, (tx, ty))
            else:
                # Each pip is a small circle. We'll compute its position relative to the
                # die's rectangle
                pip_radius = size // 10
                if value not in DICE_PIPS:
                    return  # Guard against invalid dice value
                for fx, fy in DICE_PIPS[value]:
                    px = left + int(fx * size)
                    py = top + int(fy * size)
                    pygame.draw.circle(screen, BLACK, (px, py), pip_radius)

        # Basic layout: 2 dice for the dealer on top row, 2 dice for the player on
        # bottom row.
        # We'll place them in the center horizontally, spaced horizontally as well.
        # For example, top row y=100, bottom row y=300, and x positions around 200 & 400
        # Adjust as needed.

        running = True
        while running:
            screen.fill(GREEN)  # Greenish background

            # Define font
            font = pygame.font.Font(None, 48)

            # Title text (optional)
            dealer_text = font.render("Dealer", True, WHITE)
            player_text = font.render("Player", True, WHITE)
            screen.blit(
                dealer_text, (window_width // 2 - dealer_text.get_width() // 2, 10)
            )
            screen.blit(
                player_text,
                (window_width // 2 - player_text.get_width() // 2, window_height - 50),
            )

            # Write total sum
            dealer_sum = font.render(f"{self.dealer_state[1:3].sum():02d}", True, WHITE)
            player_sum = font.render(f"{self.player_state[1:3].sum():02d}", True, WHITE)
            screen.blit(dealer_sum, (window_width // 2 - 18, 110))
            screen.blit(player_sum, (window_width // 2 - 18, window_height - 140))

            # Wrire current score
            score_text = font.render("Score:", True, WHITE)
            dealer_score = font.render(f"{self.dealer_state[0]}", True, WHITE)
            player_score = font.render(f"{self.player_state[0]}", True, WHITE)
            screen.blit(score_text, (15, 20))
            screen.blit(dealer_score, (55, 55))
            screen.blit(score_text, (15, window_height - 90))
            screen.blit(player_score, (55, window_height - 55))

            # Draw the four dice
            # Dealer dice on top
            draw_die(
                screen,
                window_width // 2 - 100,
                window_height // 2 - 75,
                self.dealer_state[1],
                size=100,
            )
            draw_die(
                screen,
                window_width // 2 + 100,
                window_height // 2 - 75,
                self.dealer_state[2],
                size=100,
            )
            # Player dice on bottom
            draw_die(
                screen,
                window_width // 2 - 100,
                window_height // 2 + 75,
                self.player_state[1],
                size=100,
            )
            draw_die(
                screen,
                window_width // 2 + 100,
                window_height // 2 + 75,
                self.player_state[2],
                size=100,
            )
            # Line between player and dealer
            pygame.draw.line(
                screen,
                (255, 0, 0),
                (0, window_height // 2),
                (window_width, window_height // 2),
                10,
            )

            pygame.display.flip()

            # Wait for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()


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
            self._state_history.append(np.append(self._state, DICE_SUM))
        return self._state_history, self.total_score

    def reset(self):
        """Reset the dealer's internal state."""
        self._state = DEFAULT_STATE.copy()
        self._roll = 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format="[%(levelname)s] %(message)s")
    logging.logProcesses = False

    prompt = (
        "Available actions:\n0 - hit first die;   1 - hit second die;   2 - hit sum;\n"
        "3 - stack first die; 4 - stack second die; 5 - stack sum.\nEnter the action: "
    )
    env = DiceBlackJack()
    state, reward, done = env.reset()
    print(f"Initial state: {state}, reward: {reward}, done: {done}")
    env.render()

    while not done:
        action = int(input(prompt))
        state, reward, done = env.step(action)
        print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
        env.render()
