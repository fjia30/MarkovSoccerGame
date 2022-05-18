from soccer_learning.environments import SoccerEnv
from abc import ABC, abstractmethod


class SoccerGameAgent(ABC):
    """Abstract class to represent agents."""

    def __init__(self, env: SoccerEnv, gamma):
        self.env = env
        self.gamma = gamma

    @abstractmethod
    def act(self, s0, s1, s2):
        pass

    @abstractmethod
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
