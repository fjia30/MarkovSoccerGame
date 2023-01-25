from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, a_action, b_action):
        pass

    @abstractmethod
    def render(self):
        pass
