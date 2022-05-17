from collections import deque
import numpy as np


class SoccerGame:
    """This is where the game is actually played.

    It takes the soccer environment, the agent and the opponent as parameters,
    and simulate a soccer game, where the agent and the opponent play against
    each other. The agent and opponent can use any of the algorithms
    implemented here, and they learn and behave independent of each other.
    The learning parameters are provided and are the same for both players."""

    def __init__(
        self,
        numEpisode,
        alpha_start,
        alpha_decay,
        alpha_min,
        epsilon_start,
        epsilon_decay,
        epsilon_min,
        gamma,
        env,
        agent,
        opponent,
        maxStep=500,
    ):
        self.alpha_start = alpha_start
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.numEpisode = numEpisode
        self.gamma = gamma
        self.env = env
        self.agent = agent
        self.opponent = opponent
        self.maxStep = maxStep

    # Sample a fixed point in agent's Q function space
    # By default, the start position.
    def __sampleAgentQValue(self, s0=2, s1=1, s2=0, a=1, o=4):
        # Special case for Q-learning
        if len(self.agent.Q.shape) < 5:
            return self.agent.Q[s0, s1, s2, a]
        return self.agent.Q[s0, s1, s2, a, o]

    def train(self):
        count = 0
        error = []
        current_val = self.__sampleAgentQValue()
        alpha = self.alpha_start

        # epsilon defines a unified exploration rate during training for both
        # players
        epsilon = self.epsilon_start

        memory = deque(maxlen=100)

        for episode in range(self.numEpisode):
            n = 1000
            if episode % n == n - 1:
                print(
                    "episode: {} / {}, "
                    "win rate={:.2f}, "
                    "alpha={:.4f}, "
                    "epsilon={:4f}".format(
                        episode,
                        self.numEpisode,
                        np.average(memory),
                        alpha,
                        epsilon,
                    )
                )
            s = self.env.reset()
            step = 0
            while True:
                if np.random.random() < epsilon:
                    agentAct = np.random.randint(self.env.action_space)
                else:
                    agentAct = self.agent.act(s[0], s[1], s[2])
                if np.random.random() < epsilon:
                    opponentAct = np.random.randint(self.env.action_space)
                else:
                    opponentAct = self.opponent.act(s[0], s[1], s[2])
                if (s[0], s[1], s[2], agentAct, opponentAct) == (
                    2,
                    1,
                    False,
                    1,
                    4,
                ):
                    count += 1
                s_prime, reward, done = self.env.step(agentAct, opponentAct)
                self.agent.learn(
                    alpha,
                    s[0],
                    s[1],
                    s[2],
                    agentAct,
                    opponentAct,
                    s_prime[0],
                    s_prime[1],
                    s_prime[2],
                    reward,
                    -reward,
                    done,
                )
                self.opponent.learn(
                    alpha,
                    s[0],
                    s[1],
                    s[2],
                    opponentAct,
                    agentAct,
                    s_prime[0],
                    s_prime[1],
                    s_prime[2],
                    -reward,
                    reward,
                    done,
                )
                if done or step > self.maxStep:
                    memory.append(reward == 100)
                    break
                s = s_prime
                step += 1
            if alpha > self.alpha_min:
                alpha *= self.alpha_decay
            if epsilon > self.epsilon_min:
                epsilon *= self.epsilon_decay
            new_val = self.__sampleAgentQValue()
            error.append(abs(new_val - current_val))
            current_val = new_val
        print(count)
        return error

    def play(self, render=True):
        s = self.env.reset()
        step = 0
        if render:
            self.env.render()
        while True:
            agentAct = self.agent.act(s[0], s[1], s[2])
            opponentAct = self.opponent.act(s[0], s[1], s[2])
            s_prime, reward, done = self.env.step(agentAct, opponentAct)
            if render:
                print("\n", agentAct, opponentAct)
                self.env.render()
            if done or step > self.maxStep:
                break
            s = s_prime
            step += 1
        return reward

    def evaluate(self, num=10000):
        rewards = []
        for i in range(num):
            rewards.append(self.play(False) == 100)
        return np.average(rewards)
