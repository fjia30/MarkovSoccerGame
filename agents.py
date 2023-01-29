import copy
import numpy as np
from cvxopt import matrix, solvers
from soccer import SoccerEnviroment
from abc import ABC, abstractmethod

# this is the interface for all agents
class ISoccerGameAgent(ABC):
    def __init__(self, env: SoccerEnviroment, gamma):
        self.env = env
        self.gamma = gamma
    
    @abstractmethod
    def act(self, s0, s1, s2):
        pass

    @abstractmethod
    def learn(self, alpha, s0, s1, s2, action, opponentAction, s_prime0, s_prime1, s_prime2, reward, opponent_reward, done):
        pass