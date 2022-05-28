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
        num_episode,
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
        max_step=500,
    ):
        self.alpha_start = alpha_start
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.numEpisode = num_episode
        self.gamma = gamma
        self.env = env
        self.agent = agent
        self.opponent = opponent
        self.maxStep = max_step

    # Sample a fixed point in agent's Q function space
    # By default, the start position.
    def _sample_agent_q_value(self, s0=2, s1=1, s2=0, a=1, o=4):
        # Special case for Q-learning
        if len(self.agent.Q.shape) < 5:
            return self.agent.Q[s0, s1, s2, a]
        return self.agent.Q[s0, s1, s2, a, o]

    def train(self):
        count = 0
        error = []
        current_val = self._sample_agent_q_value()
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
                    agent_action = np.random.randint(self.env.num_actions)
                else:
                    agent_action = self.agent.act(s[0], s[1], s[2])
                if np.random.random() < epsilon:
                    opponent_action = np.random.randint(self.env.num_actions)
                else:
                    opponent_action = self.opponent.act(s[0], s[1], s[2])
                if (s[0], s[1], s[2], agent_action, opponent_action) == (
                    2,
                    1,
                    False,
                    1,
                    4,
                ):
                    count += 1
                s_prime, reward, done = self.env.step(
                    agent_action, opponent_action
                )
                self.agent.learn(
                    alpha,
                    s[0],
                    s[1],
                    s[2],
                    agent_action,
                    opponent_action,
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
                    opponent_action,
                    agent_action,
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
            new_val = self._sample_agent_q_value()
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
            agent_action = self.agent.act(s[0], s[1], s[2])
            opponent_action = self.opponent.act(s[0], s[1], s[2])
            s_prime, reward, done = self.env.step(
                agent_action, opponent_action
            )
            if render:
                print("\n", agent_action, opponent_action)
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
