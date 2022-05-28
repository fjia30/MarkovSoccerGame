from soccer_learning.agents import SoccerGameAgent
import numpy as np


class RandomAgent(SoccerGameAgent):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)

        state_space = env.state_space
        action_space = env.num_actions

        q_dim = np.concatenate((state_space, [action_space, action_space]))

        self.Q = np.ones(q_dim)

    def act(self, s0, s1, s2):
        return np.random.randint(self.env.num_actions)

    def learn(
        self,
        alpha,
        s0,
        s1,
        s2,
        action,
        opponent_action,
        s_prime0,
        s_prime1,
        s_prime2,
        reward,
        opponent_reward,
        done,
    ):
        pass
