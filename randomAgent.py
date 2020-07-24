from agents import *

class randomPlayAgent(ISoccerGameAgent):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        stateSpace = env.state_space
        actSpace = env.action_space
        dimOfQ = np.concatenate((stateSpace, [actSpace, actSpace]))
        self.Q = np.ones(dimOfQ)

    def act(self, s0, s1, s2):
        return np.random.randint(self.env.action_space)

    def learn(self, alpha, s0, s1, s2, action, opponentAction, s_prime0, s_prime1, s_prime2, reward, opponent_reward,done):
        pass