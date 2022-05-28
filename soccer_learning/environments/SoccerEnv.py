from soccer_learning.environments import Env
import numpy as np


class SoccerEnv(Env):
    """This class implements a very simple grid-like soccer (aka football)
    environment.

    The soccer grid is a 2x4 grid. The cells in the grid are numbered in the
    following way

    +---+---+---+---+
    | 0 | 1 | 2 | 3 |
    +---+---+---+---+
    | 4 | 5 | 6 | 7 |
    +---+---+---+---+

    A state is a tuple S = (a_pos, b_pos, a_has_ball), where

    - a_pos is the position of A (which can be an integer from 0 to 7),
    - b_pos is the position of B,
    - a_has_ball is a boolean flag that indicates whether A has the ball or
    not.

    The initial position of A and B can either be 1, 2, 5 or 6, but A and B
    cannot be in the same position; so, e.g., A and B cannot be in position 2
    at the same time - if A is in position 2, then B must be initialised to be
    in either 1, 5 or 6.

    The actions are

    - north (N)
    - south (S)
    - east (E)
    - west (W)
    - stick

    This environment has 3 public methods (that every Gym environment also
    has):

        - reset(): randomly initialise the current state of the environment and
        returns it.

        - step(a_action, b_action): take a step in the environment given the
         actions from player A and player B; it returns a tuple that contains
            - the next state,
            - reward, and
            - a boolean flag that indicates whether the episode has terminated
            or not.

        - render(): draw the current state.

    Note that the step method receives 2 parameters in our case, because this
    environment has 2 agents, each of which can take an action at every step.
    """

    GOAL_REWARD = 100

    def __init__(self):
        super().__init__()

        self.actions = [-4, 4, 1, -1, 0]

        self.num_actions = len(self.actions)

        # 8 ways to place A, 7 remaining ways to place B (because B cannot be
        # placed in the same cell of A), and 2 ways to place the ball - either
        # give it to A or B.
        # self.num_states = 8 * 7 * 2

        self.positions = list(range(8))

        # (number of positions for player A, number of positions for player B,
        # number of positions for the ball)
        self.state_space = (8, 8, 2)

        self.a_pos = None
        self.b_pos = None
        self.a_has_ball = None

        # Call reset to initialise a_pos, b_pos and a_has_ball.
        self.reset()

    @property
    def current_state(self):
        return self.a_pos, self.b_pos, self.a_has_ball

    def _calculate_reward(self):
        """Calculates the reward for player A.

        The reward for B is the negative of the reward for A, by definition of
        a zero-sum game.

        :return: the reward for A."""
        if self.a_has_ball:
            if self.a_pos == 0 or self.a_pos == 4:
                return SoccerEnv.GOAL_REWARD
            if self.a_pos == 3 or self.a_pos == 7:
                return -SoccerEnv.GOAL_REWARD
        else:
            if self.b_pos == 0 or self.b_pos == 4:
                return SoccerEnv.GOAL_REWARD
            if self.b_pos == 3 or self.b_pos == 7:
                return -SoccerEnv.GOAL_REWARD
        return 0

    def _move_player(self, position, action):
        """Calculate the new position of a player after taking the action in
        position.

        If the action leads to a position outside the soccer grid, the original
        position (before taking the `action`) is returned.

        :param position: the position of the player
        :param action: the action to be taken from the position of the player.
        :return: the new position of the player after taking the action."""
        assert position in self.positions
        new_position = position + self.actions[action]
        if new_position not in self.positions:
            return position
        else:
            return new_position

    def _move_a(self, a_action):
        new_a_pos = self._move_player(self.a_pos, a_action)
        if new_a_pos != self.b_pos:
            # Only assign the new position if it's different from self.b_pos.
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
        self.a_pos, self.b_pos = np.random.choice(
            [1, 2, 5, 6], size=2, replace=False
        )
        assert all(x in self.positions for x in [self.a_pos, self.b_pos])
        self.a_has_ball = np.random.choice([True, False])
        return self.current_state

    def step(self, a_action, b_action):
        """Take a step in the game, given the actions of A and B.

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

        return self.current_state, reward, not reward == 0

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


if __name__ == "__main__":
    e = SoccerEnv()
    e.render()
