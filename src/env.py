"""Module for DiceBlackJack environment and associated Dealer class."""

import logging
from collections.abc import Callable
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame

DEFAULT_STATE = np.array([0, 0, 0, 0], dtype=np.uint8)

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

    def __init__(self, dealer_th: int = 17, render_mode: Optional[str] = None):
        """Initialize the environment."""
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.MultiDiscrete([25, 7, 7, 2])
        self.dealer = Dealer(self._roll_dice, dealer_th)
        self.render_mode = render_mode
        logger.info("Dice Blackjack environment has been initialized.")
        self.done = True

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int, bool]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.dealer.reset()
        self.player_state = DEFAULT_STATE.copy()
        self.player_score = 0
        self.dealer_state = DEFAULT_STATE.copy()
        self.dealer_score = 0
        self._chance_for_blackjack = False
        self._blackjack = False
        self.done = False

        self.player_state[1:3] = self._roll_dice()
        self._check_blackjack()
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

        is_double_score = self.player_state[3] == 0
        die_idx = 2 if is_double_score else action % 3
        is_stack = action // 3

        # Record state for debug purpose
        previous_state = self.player_state.copy()

        if is_stack:
            # Add points to the score
            dice = self.player_state[1:3]
            dice = np.append(dice, dice.sum())
            if is_double_score:
                self.player_score = 2 * (self.player_state[0] + dice[die_idx])
            else:
                self.player_score += dice[die_idx]
            # Engage dealer
            self.dealer_state, self.dealer_score = self.dealer.play()
            self.done = True
            # Get the observation
            logger.debug("Transition %s -> %s", previous_state, self.player_state)
            return self._get_observation()

        else:
            # Add points to the score
            dice = self.player_state[1:3]
            dice = np.append(dice, dice.sum())
            if is_double_score:
                self.player_state[0] += 2 * dice[die_idx]
            else:
                self.player_state[0] += dice[die_idx]
            self.player_score = self.player_state[0]
            self.player_state[3] = 1  # It is not the first roll
            # Roll the dice!
            dice = self._roll_dice()
            self.player_state[1:3] = dice
            # Get the observation
            self._check_blackjack()
            logger.debug("Transition %s -> %s", previous_state, self.player_state)
            return self._get_observation()

    def _check_blackjack(self) -> None:
        """Check the Blackjack combination."""
        if self.player_state[1] == self.player_state[2] and self._chance_for_blackjack:
            self._blackjack = True
        elif self.player_state[1] == self.player_state[2]:
            self._chance_for_blackjack = True
            self._blackjack = False
        else:
            self._chance_for_blackjack = False
            self._blackjack = False

    def _get_observation(self) -> None:
        self._update_players_score()
        reward = self._calculate_reward()
        logger.debug(
            "State: %s, reward: %d, is done: %s",
            str(self.player_state),
            reward,
            self.done,
        )
        return self.player_state, reward, self.done

    def _update_players_score(self) -> None:
        """Recalculate the player score if the bust is unavoidable."""
        min_score = 0
        if not self._chance_for_blackjack and self.player_state[3] > 0:
            # Enable if there is no chance for Blackjack
            min_score = self.player_state[0] + self.player_state[1:3].min()
        elif not self._chance_for_blackjack and self.player_state[3] == 0:
            min_score = 2 * self.player_state[1:3].sum()

        if min_score > BUST_VALUE:
            self.player_score = min_score

    def _calculate_reward(self) -> int:
        """Reward function implementation."""
        if self._blackjack:
            self.done = True
            logger.debug(
                "Game ended with the Blackjack combination!"
            )
            return 1  # Ultimate victory
        elif self.dealer_score > BUST_VALUE:
            logger.debug(
                "Game ended with the dealer's bust of score %02d", self.dealer_score
            )
            self.done = True
            return 1  # Dealer's bust
        elif self.player_score > BUST_VALUE:
            logger.debug(
                "Game ended with the player's bust of score %02d", self.player_score
            )
            self.done = True
            return -1  # Bust lose
        elif self.player_score > self.dealer_score and self.done:
            logger.debug(
                "Game ended with the player's victory: %02d > %02d",
                self.player_score,
                self.dealer_score,
            )
            return 1  # Player's victory
        elif self.player_score < self.dealer_score and self.done:
            logger.debug(
                "Game ended with the dealer's victory: %02d < %02d",
                self.player_score,
                self.dealer_score,
            )
            self.done = True
            return -1  # Score lose
        elif self.player_score == self.dealer_score and self.done:
            logger.debug(
                "Game ended with the draw: %02d = %02d",
                self.player_score,
                self.dealer_score,
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
            import gymnasium as gym
            gym.logger.warn(
                "You are calling render() without specifying a render mode. "
                'You can specify it at initialization, e.g. gym.make("DiceBlackjack-v0", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise Exception('pygame is not installed. Run `pip install pygame`') from e

        # Window dimensions
        window_width, window_height = 600, 400

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

        # Fill the background
        self.screen.fill(GREEN)

        # --- Helper function to draw a single die ---
        def draw_die(surface, center_x, center_y, value, size=100):
            """Draw a single die on the given surface."""
            # Compute the top-left coordinates of the die's rectangle
            left = center_x - size // 2
            top = center_y - size // 2
            # Draw die background with rounded corners
            pygame.draw.rect(surface, WHITE, (left, top, size, size), border_radius=10)
            pygame.draw.rect(surface, BLACK, (left, top, size, size), 2, border_radius=10)

            if value is None:
                # Draw a question mark if value is missing
                font = pygame.font.SysFont(None, size)
                text_surface = font.render("?", True, BLACK)
                tx = center_x - text_surface.get_width() // 2
                ty = center_y - text_surface.get_height() // 2
                surface.blit(text_surface, (tx, ty))
            else:
                # Draw the pips based on the provided mapping in DICE_PIPS
                pip_radius = size // 10
                if value not in DICE_PIPS:
                    return  # Guard against an invalid dice value
                for fx, fy in DICE_PIPS[value]:
                    px = left + int(fx * size)
                    py = top + int(fy * size)
                    pygame.draw.circle(surface, BLACK, (px, py), pip_radius)

        # --- Rendering text and dice layout ---
        # Define a font (you may choose a different size or font file)
        font = pygame.font.Font(None, 48)

        # Render and position dealer and player labels
        dealer_text = font.render("Dealer", True, WHITE)
        player_text = font.render("Player", True, WHITE)
        self.screen.blit(dealer_text, (window_width // 2 - dealer_text.get_width() // 2, 10))
        self.screen.blit(
            player_text,
            (window_width // 2 - player_text.get_width() // 2, window_height - 50),
        )

        # Compute and render dice sums (assumes self.dealer_state and self.player_state are sequences)
        dealer_sum_value = sum(self.dealer_state[1:3])
        player_sum_value = sum(self.player_state[1:3])
        dealer_sum = font.render(f"{dealer_sum_value:02d}", True, WHITE)
        player_sum = font.render(f"{player_sum_value:02d}", True, WHITE)
        self.screen.blit(dealer_sum, (window_width // 2 - 18, 110))
        self.screen.blit(player_sum, (window_width // 2 - 18, window_height - 140))

        # Render scores (assumes the first element holds the current score)
        score_text = font.render("Score:", True, WHITE)
        dealer_score = font.render(f"{self.dealer_state[0]}", True, WHITE)
        player_score = font.render(f"{self.player_state[0]}", True, WHITE)
        self.screen.blit(score_text, (15, 20))
        self.screen.blit(dealer_score, (55, 55))
        self.screen.blit(score_text, (15, window_height - 90))
        self.screen.blit(player_score, (55, window_height - 55))

        # Draw the four dice:
        # Dealer's dice (top row)
        draw_die(
            self.screen,
            window_width // 2 - 100,
            window_height // 2 - 75,
            self.dealer_state[1],
            size=100,
        )
        draw_die(
            self.screen,
            window_width // 2 + 100,
            window_height // 2 - 75,
            self.dealer_state[2],
            size=100,
        )
        # Player's dice (bottom row)
        draw_die(
            self.screen,
            window_width // 2 - 100,
            window_height // 2 + 75,
            self.player_state[1],
            size=100,
        )
        draw_die(
            self.screen,
            window_width // 2 + 100,
            window_height // 2 + 75,
            self.player_state[2],
            size=100,
        )

        # Draw a dividing line between dealer and player areas
        pygame.draw.line(
            self.screen,
            (255, 0, 0),
            (0, window_height // 2),
            (window_width, window_height // 2),
            10,
        )

        # --- Finalize the frame based on render mode ---
        if self.render_mode == "human":
            pygame.event.pump()  # Process event queue
            pygame.display.flip()
            # Use the specified render fps (defaulting to 30 if not in metadata)
            self.clock.tick(self.metadata.get("render_fps", 30))
        else:
            # For non-human render mode, return an RGB array of the frame.
            import numpy as np
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


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
        self._roll = 1
        self.total_score = 2 * dice.sum()  # First roll
        self._state[1:3] = dice

        while self.total_score < self._threshold:
            self._state[0] = self.total_score
            # Roll the dice
            dice = self._roll_generator()
            self.total_score += dice.sum()
            self._state[1:3] = dice
            # If the roll is not first, change the value
            self._roll += 1
        self._state[3] = int(self._roll > 1)
        return self._state, self.total_score

    def reset(self):
        """Reset the dealer's internal state."""
        self._state = DEFAULT_STATE.copy()
        self._roll = 0


if __name__ == "__main__":
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    std_format = logging.Formatter("[%(levelname)s] %(message)s")
    stream_handler.setFormatter(std_format)

    logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler])
    logging.logProcesses = False

    prompt = (
        "Available actions:\n0 - hit first die;   1 - hit second die;   2 - hit sum;\n"
        "3 - stack first die; 4 - stack second die; 5 - stack sum.\nEnter the action: "
    )
    env = DiceBlackJack(render_mode='rgb')
    state, reward, done = env.reset()
    print(f"Initial state: {state}, reward: {reward}, done: {done}")
    env.render()

    while not done:
        action = int(input(prompt))
        state, reward, done = env.step(action)
        print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
        env.render()

    # Final render call (optional)
    env.render()
    
    # Add a loop to keep the window open until the user closes it.
    if env.render_mode == "human":
        print("Game over. Close the window or press Ctrl+C to exit.")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
