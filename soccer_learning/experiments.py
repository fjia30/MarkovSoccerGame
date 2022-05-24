from soccer_learning.environments import SoccerEnv
from soccer_learning.agents import RandomAgent
from soccer_learning.agents import QLearning
from soccer_learning.agents import FoeQ
from soccer_learning.agents import FriendQ
from soccer_learning.agents import CEQ
from matplotlib import pyplot as plt
from soccer_learning.games import SoccerGame
import itertools


def main():
    num_episode = 100000
    alpha_start = 1
    alpha_decay = 0.99993
    alpha_min = 0.001
    epsilon_start = 1
    epsilon_decay = 0.99993
    epsilon_min = 0.01
    gamma = 0.99

    env = SoccerEnv()

    agents_classes = [QLearning, FoeQ, FriendQ, CEQ, RandomAgent]

    trained_agents = {}

    for agent_class in agents_classes:
        agent = agent_class(env, gamma)
        opponent = agent_class(env, gamma)

        game = SoccerGame(
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
        )

        error = game.train()

        trained_agents[agent] = "Agent"
        trained_agents[opponent] = "Opponent"

        title = "Error ({} vs {})".format(
            agent.__class__.__name__, opponent.__class__.__name__
        )

        plt.figure()
        plt.plot(error, linewidth=0.5)
        plt.title(title)
        plt.show()

        # plt.figure()
        # plt.plot(error, linewidth=0.5)
        # plt.title(title)
        # plt.ylim(0, 0.01)
        # plt.show()

    combinations = itertools.combinations(trained_agents, r=2)
    for agent, opponent in combinations:
        game = SoccerGame(
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
        )

        print(
            "Evaluation ({} [{}] vs {} [{}]) = {}".format(
                agent.__class__.__name__,
                trained_agents[agent],
                opponent.__class__.__name__,
                trained_agents[opponent],
                game.evaluate(),
            )
        )


if __name__ == "__main__":
    main()
