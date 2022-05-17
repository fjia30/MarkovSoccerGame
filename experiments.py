from SoccerEnv import SoccerEnv
from RandomAgent import RandomAgent
from QLearningAgent import QLearning
from FoeQ import FoeQ
from FriendQ import FriendQ
from CEQ import CEQ
from matplotlib import pyplot as plt
from SoccerGame import SoccerGame

num_episode = 100000
epsilon_start = 1
epsilon_decay = 0.99993
epsilon_min = 0.01
gamma = 0.99
alpha_start = 1
alpha_decay = 0.99993
alpha_min = 0.001
env = SoccerEnv()

q_learning_agent = QLearning(env, gamma)
q_learning_opponent = QLearning(env, gamma)
game1 = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    q_learning_agent,
    q_learning_opponent,
)
q_learning_error = game1.train()
plt.plot(q_learning_error, linewidth=0.5)
plt.show()

plt.plot(q_learning_error, linewidth=0.5)
plt.ylim(0, 0.01)

foe_q_agent = FoeQ(env, gamma)
foe_q_opponent = FoeQ(env, gamma)
game2 = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    foe_q_agent,
    foe_q_opponent,
)
foe_q_error = game2.train()
plt.plot(foe_q_error, linewidth=0.5)
plt.show()

plt.plot(foe_q_error, linewidth=0.5)
plt.ylim(0, 0.01)

friend_q_agent = FriendQ(env, gamma)
friend_q_opponent = FriendQ(env, gamma)
game3 = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    friend_q_agent,
    friend_q_opponent,
)
friend_q_error = game3.train()
plt.plot(friend_q_error, linewidth=0.5)
plt.show()

plt.plot(friend_q_error, linewidth=0.5)
plt.ylim(0, 0.01)

ceq_agent = CEQ(env, gamma)
ceq_opponent = CEQ(env, gamma)
game4 = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    ceq_agent,
    ceq_opponent,
)
ceq_error = game4.train()
plt.plot(ceq_error, linewidth=0.5)
plt.show()

plt.plot(ceq_error, linewidth=0.5)
plt.ylim(0, 0.01)

ceq_vs_foe = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    ceq_agent,
    foe_q_opponent,
)
print(ceq_vs_foe.evaluate())

ceq_vs_friend = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    ceq_agent,
    friend_q_opponent,
)
print(ceq_vs_friend.evaluate())

foe_vs_friend = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    foe_q_agent,
    friend_q_opponent,
)
print(foe_vs_friend.evaluate())

friend_vs_foe = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    friend_q_agent,
    foe_q_opponent,
)
print(friend_vs_foe.evaluate())

ceq_vs_q_learning = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    ceq_agent,
    q_learning_opponent,
)
print(ceq_vs_q_learning.evaluate())

foe_q_vs_q_learning = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    foe_q_agent,
    q_learning_opponent,
)
print(foe_q_vs_q_learning.evaluate())

friend_q_vs_q_learning = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    friend_q_agent,
    q_learning_opponent,
)
print(friend_q_vs_q_learning.evaluate())

random_agent = RandomAgent(env, gamma)
ceq_vs_random = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    ceq_agent,
    random_agent,
)
print("CEQ vs Random: ", ceq_vs_random.evaluate())
foe_q_vs_random = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    foe_q_agent,
    random_agent,
)
print("FoeQ vs Random: ", foe_q_vs_random.evaluate())
friend_q_vs_random = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    friend_q_agent,
    random_agent,
)
print("FriendQ vs Random: ", friend_q_vs_random.evaluate())
q_learning_vs_random = SoccerGame(
    num_episode,
    alpha_start,
    alpha_decay,
    alpha_min,
    epsilon_start,
    epsilon_decay,
    epsilon_min,
    gamma,
    env,
    q_learning_agent,
    random_agent,
)
print("QLearn vs Random: ", q_learning_vs_random.evaluate())
