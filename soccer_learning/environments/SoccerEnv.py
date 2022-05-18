from soccer_learning.environments import Env
import numpy as np

GOAL_REWARD = 100


class SoccerEnv(Env):
    """This represents the soccer environment.

    It is designed to be similar to OpenAI's gym environments.

    Use reset() to initiate an episode.

    Use step(actionA, actionB) to simulate an action which returns next a
    state, reward and isFinished.

    Use render() to draw the current state.

    self.action_space: num of actions
    self.state_space: <num of variable1, num of variable2, num of variable3>

    The field is a 2x4 grid.

    Number the grid as

    0, 1, 2, 3
    4, 5, 6, 7

    States are the position of A, position of B and whether A or B has the ball
    actions for both A and B are (N, S, E, W, stick), which is represented as
    0~4."""

    def __init__(self):
        super().__init__()
        self.actions = [-4, 4, 1, -1, 0]
        self.action_space = len(self.actions)
        self.state_space = (8, 8, 2)

    def _show_current_state(self):
        return self.a_pos, self.b_pos, self.a_has_ball

    def _calculate_reward(self):
        """Calculates the reward for A.

        The reward for B is the negative of the reward for A, by definition of
        a zero-sum game.

        :return: the reward for A."""
        if self.a_has_ball:
            if self.a_pos == 0 or self.a_pos == 4:
                return GOAL_REWARD
            if self.a_pos == 3 or self.a_pos == 7:
                return -GOAL_REWARD
        else:
            if self.b_pos == 0 or self.b_pos == 4:
                return GOAL_REWARD
            if self.b_pos == 3 or self.b_pos == 7:
                return -GOAL_REWARD
        return 0

    def _move_player(self, position, action):
        """Calculate the position of a player after a move.

        Player sticks if moving towards a wall.

        :param position:
        :param action:
        :return:"""
        new_position = position + self.actions[action]
        if new_position < 0 or new_position > 7:
            return position
        else:
            return new_position

    def _move_a(self, a_action):
        new_a_pos = self._move_player(self.a_pos, a_action)
        if new_a_pos != self.b_pos:
            self.a_pos = new_a_pos
        elif self.a_has_ball:
            # If A run into B with a ball, give the ball to B.
            self.a_has_ball = False

    def _move_b(self, b_action):
        new_b_pos = self._move_player(self.b_pos, b_action)
        if new_b_pos != self.a_pos:
            self.b_pos = new_b_pos
        elif not self.a_has_ball:
            # If B run into A with a ball, give the ball to A.
            self.a_has_ball = True

    def reset(self):
        """Initialize the environment by giving random positions to A, B and
        the ball.

        :return: the current state."""
        # TODO: this should be defined first in the __init__ method
        self.a_pos, self.b_pos = np.random.choice(
            [1, 2, 5, 6], size=2, replace=False
        )
        self.a_has_ball = np.random.choice([True, False])
        return self._show_current_state()

    def step(self, a_action, b_action):
        """Take a step in the game, given actions of A and B.

        :param a_action: action of A
        :param b_action: action of B
        :return: return the next state, reward and whether the game is done."""
        if np.random.random() > 0.5:
            # A moves first.
            self._move_a(a_action)
            self._move_b(b_action)
        else:
            # B moves first.
            self._move_b(b_action)
            self._move_a(a_action)

        reward = self._calculate_reward()
        return self._show_current_state(), reward, not reward == 0

    def render(self):
        out = "---------------------\n"
        for i in range(2):
            for j in range(4):
                position = i * 4 + j
                if self.a_pos == position:
                    if self.a_has_ball:
                        out += "| A* "
                    else:
                        out += "| A  "
                elif self.b_pos == position:
                    if not self.a_has_ball:
                        out += "| B* "
                    else:
                        out += "| B  "
                else:
                    out += "|    "
            out += "|\n---------------------\n"
        print(out)
