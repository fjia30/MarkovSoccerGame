import numpy as np
from soccer_learning.agents import SoccerGameAgent


class QLearning(SoccerGameAgent):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)

        state_space = env.state_space
        action_space = env.num_actions

        q_dim = np.concatenate((state_space, [action_space]))

        self.Q = np.ones(q_dim)

    def act(self, s0, s1, s2):
        s2 = int(s2)

        q_values = self.Q[s0, s1, s2]

        # pick the best action, tie-break randomly
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

        return action

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
        s2 = int(s2)
        s_prime2 = int(s_prime2)
        # Step 4a: calculate v_prime for the end state, next state value is 0.
        if not done:
            v_prime = np.max(self.Q[s_prime0, s_prime1, s_prime2])
        else:
            v_prime = 0

        # Step 4b: update Q, which is the same as in FoeQ.
        # Q[s, a] = (1 - alpha) * Q[s, a] + alpha *
        # ((1 - gamma) * rew + gamma * V[s'])
        # Simple Q-learning does not consider opponent's actions.
        self.Q[s0, s1, s2, action] = (1 - alpha) * self.Q[
            s0, s1, s2, action
        ] + alpha * ((1 - self.gamma) * reward + self.gamma * v_prime)
