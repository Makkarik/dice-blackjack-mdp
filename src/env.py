"""Module for DiceBlackJack environment and associated Dealer class.

    RULES:

1.  Player roll two dices at the first step. The sum of the dices is doubled and goes to
    the initial player's score. After the first roll the player can either `hit` (roll
    dice) or `stack` (stop rolling).

2.  If player continues rolling the dice, he may choose at any step either he add to his
    sum the value of any dice or a sum of the values. His main goal is to get more
    points than dealer, but no more than 21.

3.  When player stops rolling, the dealer starts the play. Dealer must roll dices until
    he reach 17 point. The first step of the dealer gives him double score, at the
    second step and further he may decide what die to choose to get the maximum score,
    but stops as soon as scores 17 points or more.

4.  The player wins the double bid (+1 point) if he scores more than dealer, but no more
    than 21. If the player and dealer roll the same number, then the tie happens -
    neither of them get the bid (0 points). If any player scored less point than the
    dealer then he lose his bead (-1 point).

5.  If any party get more than 21 points, the opposite party wins immediately.

6.  If the player rolls two double values in a row at any step from the first roll, then
    he gets a Blackjack and immediately wins. The Blackjack happens even if player
    scored more than 21 points. The dealer can not roll the Blackjack at any case.

    For further details see: https://www.chessandpoker.com/dice_blackjack.html


    STATE SPACE:

    (25, 7, 7)

    25 - The number of scores, rolled during the previous steps.
     7 - All possible values of the die, including 0, that designates the lack of die.

    Some explanation regarding 24 options for the score.

    It is obvious that you may get up to 27 points by stacking the sum of the dice in
    the case, when it is still possible get no more than 21 points [16, 5, 6, X]. Thus,
    the maximum possible value is 27. Also, it is impossible to get scores from 1 to 3,
    as the minimal roll of dice yields 4 points at least. Therefore, the maximum number
    of score variations is 28 - 3 = 25.

    In real life most of the states are not reachable. The Q-Learning agent has
    reached 910 states only after 100.000 iterations inside the training cycle.


    REWARDS:

    -1 point  - player lost or got busted
     0 point  - player got a tie
     1 point  - player win or dealer got busted
     2 points - player roll the Blackjack


    ACTIONS:

    0 - hit the first die      3 - stack the first die
    1 - hit the second die     4 - stack the second die
    2 - hit the sum of dice    5 - stack the sum of dice

    CREDITS:

    Grand9K Pixel font - Jayvee D. Enaguas (Grand Chaos), 2013.

"""

import logging
import os
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

DEFAULT_STATE = np.array([0, 0, 0])

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

WHITE = (0xFD,) * 3


logger = logging.getLogger(__name__)


class DiceBlackJack(gym.Env):
    """Dice Blackjack RL environment implementation.

    Attributes
    ----------
    action_space : gymnasium.spaces.Discrete
        The size of the action space.
    observation_space : gymnasium.spaces.MultiDiscrete
        The size of the observation space.
    dealer : Dealer
        The instance of the dealer, used by the environment to play against the player.
    done : bool
        A flag, indicating either the game has been ended or not.

    Methods
    -------
    reset(seed: int | None = None) -> tuple[tuple, dict[str, Any]]
        The method for resetting the environment. Must be called each time the "done"
        flag has been raised.
    step(self, action: int) -> tuple[tuple, float, bool, bool, dict[str, Any]]
        The method for making a step with the chosem action. The action must be from the
        action_space
    render() -> None
        A method to render the current state of the environment.

    """

    def __init__(
        self, dealer_th: int = 17, render_mode: str | None = None, fps: int = 10
    ):
        """Initialize the environment.

        Parameters
        ----------
        dealer_th : int = 17
            The threshold, until which the dealer must roll the scores.
        render_mode : str | None = None
            The render mode. Can be either "human" or "rgb_array".
        fps : int = 10
            Frame per second limitaion for the rendering option.

        """
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.MultiDiscrete([28, 7, 7, 2])
        self.dealer = Dealer(self._roll_dice, dealer_th)
        self.render_mode = render_mode
        self.metadata["render_fps"] = fps
        self.metadata["render_modes"] = ["human", "rgb_array"]
        self.done = True

        self._cwd = os.path.dirname(__file__)
        logger.info("Dice Blackjack environment has been initialized.")

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[tuple, dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int | None = None
            A seed for the internal random generator.
        options : dict[str, Any] | None = None
            A parameter, utilized by the wrappers and other utilities.

        Returns
        -------
        observation, info : tuple[tuple, dict[str, Any]]
            The initial observation after the reset with the auxilary information.

        """
        super().reset(seed=seed, options=options)
        self.dealer.reset()

        self._player_state = DEFAULT_STATE.copy()
        self._player_history = []
        self._player_min_score = 0

        self._dealer_state = DEFAULT_STATE.copy()
        self._dealer_min_score = 0
        self._dealer_history = [np.append(self._dealer_state.copy(), None)]

        self._chance = False
        self._blackjack = False
        self.done = False

        self._player_state[1:3] = self._roll_dice()
        # Check if the double value has been rolled
        if self._player_state[1] == self._player_state[2]:
            self._chance = True
        self._player_history.append(np.append(self._player_state.copy(), None))
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

        self._player_min_score = self._get_player_score(self._player_state)
        if self._player_min_score > BUST_VALUE:
            self.done = True

        else:
            is_double_score = self._player_state[0] == 0
            # If the first roll, map actions to choosing the sum of dice
            action = 3 * (action // 3) + 2 if is_double_score else action
            die_idx = action % 3
            is_stack = action // 3

            # Record state for debug purpose
            previous_state = self._player_state.copy()

            if is_stack:
                # Add points to the score
                dice = self._player_state[1:3]
                dice = np.append(dice, dice.sum())
                self._chance = False
                if is_double_score:
                    self._player_state[0] = 2 * dice[die_idx]
                else:
                    self._player_state[0] += dice[die_idx]
                self._player_state[1:3] = 0
                self._player_min_score = self._get_player_score(self._player_state)
                if self._player_min_score <= BUST_VALUE:
                    # Engage dealer
                    self._dealer_history, self._dealer_min_score = self.dealer.play()
                    self._dealer_state = self._dealer_history[-1][:-1]
                self.done = True
                # Get the observation
                logger.debug("Transition %s -> %s", previous_state, self._player_state)

            else:
                # Add points to the score
                dice = self._player_state[1:3]
                dice = np.append(dice, dice.sum())
                if is_double_score:
                    self._player_state[0] = 2 * dice[die_idx]
                else:
                    self._player_state[0] += dice[die_idx]
                # Roll the dice!
                dice = self._roll_dice()
                # Check the blackjack
                if dice[0] == dice[1] and self._chance:
                    self._blackjack = True
                else:
                    self._chance = False
                    self._blackjack = False
                self._player_state[1:3] = dice
                self._player_min_score = self._get_player_score(self._player_state)
                logger.debug("Transition %s -> %s", previous_state, self._player_state)
            # Write action to the previous state
            self._player_history[-1][-1] = int(action)
            self._player_history.append(np.append(self._player_state.copy(), None))

        reward = self._calculate_reward()
        logger.debug(
            "State: %s, reward: %d, is done: %s",
            str(self._player_state),
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
                str(np.stack(self._player_history)).replace("\n", " ->"),
            )
            logger.info(
                "Dealer's rolls: %s",
                str(np.stack(self._dealer_history)).replace("\n", " ->"),
            )
        # Human-readable option enables default rendering
        if self.render_mode == "human":
            self.render()
        return tuple(map(int, self._player_state))

    def _get_player_score(self, state: np.ndarray) -> None:
        """Recalculate the player score."""
        if self._chance:
            min_score = 0
        elif state[0] > BUST_VALUE:
            min_score = state[0]
        elif state[0] == 0:
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
        elif self._player_min_score > BUST_VALUE:
            logger.info(
                "Game ended with the player's bust of score %02d",
                self._player_min_score,
            )
            self.done = True
            return -1.0  # Bust lose
        elif self._dealer_min_score > BUST_VALUE:
            logger.info(
                "Game ended with the dealer's bust of score %02d",
                self._dealer_min_score,
            )
            self.done = True
            return 1.0  # Dealer's bust
        elif self._player_state[0] > self._dealer_state[0] and self.done:
            logger.info(
                "Game ended with the player's victory: %02d > %02d",
                self._player_min_score,
                self._dealer_min_score,
            )
            return 1.0  # Player's victory
        elif self._player_state[0] < self._dealer_state[0] and self.done:
            logger.info(
                "Game ended with the dealer's victory: %02d < %02d",
                self._player_min_score,
                self._dealer_min_score,
            )
            self.done = True
            return -1.0  # Score lose
        elif self._player_state[0] == self._dealer_state[0] and self.done:
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
        elif self.render_mode not in self.metadata["render_modes"]:
            logger.warning('The render mode must be either "human" or "rgb_array"')
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

        player_history = truncate_list(self._player_history)
        for i, roll in enumerate(player_history):
            self._draw_roll(self.screen, 139 + 120 * i, 360, roll)

        dealer_history = truncate_list(self._dealer_history)
        for i, roll in enumerate(dealer_history):
            self._draw_roll(self.screen, 139 + 120 * i, 36, roll)

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
        if state[-1] is not None and state[-1] % 3 in {0, 2}:
            surface.blit(selection, (left + 6, top + 78))
        if state[-1] is not None and state[-1] % 3 in {1, 2}:
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
        # The dealer uses the environment's random number generator
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
            dice = self._roll_generator()

            # The dealer is working by utilizing the decision tree with 6 joints. Its
            # main goal is to maximize the total number of points, but stop after
            # hitting 17 and not get busted.

            # Firstly, the dealer check if it is possible to use both dice
            if self.total_score + dice.sum() <= BUST_VALUE:
                self.total_score += dice.sum()
                action = self._hit_or_stack() + ACTIONS["hit_sum"]

            # Then the use of the greater die is considered
            elif self.total_score + dice.max() <= BUST_VALUE:
                self.total_score += dice.max()
                action = self._hit_or_stack() + dice.argmax()

            # After that, the die with the smaller value is considered
            elif self.total_score + dice.min() <= BUST_VALUE:
                self.total_score += dice.min()
                action = self._hit_or_stack() + dice.argmin()

            # Lastly, there is no way but to choose the smaller die and get the bust
            else:
                self.total_score += dice.min()
            self._state[1:3] = dice
            self._roll += 1
            # Append state to the log
            self._state_history.append(np.append(self._state, action))

        if self.total_score <= BUST_VALUE:
            # Get to the save terminating state
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


if __name__ == "__main__":
    """
    The following snippet has been added for manula debugging and testing purposes.

    To try a game, use the keys from 1 to 6. The game window will appear as soon as you
    launch the current file.
    """
    import sys

    # Choose the apropriate logging level
    logging.basicConfig(
        level=logging.DEBUG, stream=sys.stdout, format="[%(levelname)s] %(message)s"
    )

    prompt = (
        "Available actions:\n"
        "| 1 - hit first die   | 2 - hit second die   | 3 - hit sum   |\n"
        "| 4 - stack first die | 5 - stack second die | 6 - stack sum |\n"
        "Enter the action: "
    )
    # Init the environment
    env = DiceBlackJack(render_mode="human", fps=25)
    while True:
        state, _ = env.reset()
        done = False
        logger.info("Initial state: %s", state)
        if env.render_mode != "human":
            logger.debug("Output array: %s", env.render().shape)
        # Launch the while loop until the game is finished
        while not done:
            action = int(input(prompt)) - 1  # Map actions [1, 6] => [0, 5]
            state, reward, done, _, _ = env.step(action)
            logger.info(
                "Action: %s, State: %s, Reward: %s, Done: %s",
                action,
                state,
                reward,
                done,
            )
            if env.render_mode != "human":
                logger.debug("Output array: %s", env.render().shape)
            if reward == 2:  # noqa: PLR2004
                print("Blackjack!")
            if reward < 0:
                print("Player has lost")
            elif reward > 0:
                print("Player has won")
            else:
                print("Gane has ended with tie")

        if input("Game ended. Enter '0' to exit or any key to retry: ") == "0":
            break

    pygame.quit()
