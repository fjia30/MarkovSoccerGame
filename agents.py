import copy
import numpy as np
from cvxopt import matrix, solvers
from soccer import SoccerEnviroment

# this is the interface for all agents
class SoccerGameAgent:
    def __init__(self, env: SoccerEnviroment, gamma):
        self.env = env
        self.gamma = gamma
    # def act(self, s0, s1, s2)->action:
    # def learn(self, alpha, s0, s1, s2, action, opponentAction, s_prime0, s_prime1, s_prime2, reward, opponent_reward, done):