import numpy as np
from agents import ISoccerGameAgent
from soccer import SoccerEnviroment


class QLearning(ISoccerGameAgent):
    def __init__(self, env: SoccerEnviroment, gamma):
        self.env = env
        self.gamma = gamma
        stateSpace = env.state_space
        actSpace = env.action_space
        dimOfQ = np.concatenate((stateSpace, [actSpace]))
        self.Q = np.ones(dimOfQ)

    def act(self, s0, s1, s2):
        s2 = int(s2)
        Qvalues = self.Q[s0, s1, s2]
        # pick the best action, tie-break randomly
        action = np.random.choice(np.flatnonzero(Qvalues == Qvalues.max()))
        return action

    def learn(
        self,
        alpha,
        s0,
        s1,
        s2,
        action,
        opponentAction,
        s_prime0,
        s_prime1,
        s_prime2,
        reward,
        opponent_reward,
        done,
    ):
        s2 = int(s2)
        s_prime2 = int(s_prime2)
        # Step 4a: calculate V_prime for the end state, next state value is 0.
        if not done:
            V_prime = np.max(self.Q[s_prime0, s_prime1, s_prime2])
        else:
            V_prime = 0

        # Step 4b: update Q, which is the same as in FoeQ.
        # Q[s, a] = (1 - alpha) * Q[s, a] + alpha *
        # ((1 - gamma) * rew + gamma * V[s'])
        # Simple Q-learning does not consider opponent's actions.
        self.Q[s0, s1, s2, action] = (1 - alpha) * self.Q[
            s0, s1, s2, action
        ] + alpha * ((1 - self.gamma) * reward + self.gamma * V_prime)
        pass
