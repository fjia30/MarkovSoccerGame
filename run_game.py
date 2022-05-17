from soccer import SoccerEnviroment
from randomAgent import randomPlayAgent
from QlearningAgent import QLearning
from foeQ import FoeQ
from friendQ import FriendQ
from ceQ import CEQ
from matplotlib import pyplot as plt
from game_interface import SoccerGame

numEpisode = 100000
epsilon_start = 1
epsilon_decay = 0.99993
epsilon_min = 0.01
gamma = 0.99
alpha_start = 1
alpha_decay = 0.99993
alpha_min = 0.001
env = SoccerEnviroment()

QLearnAgent = QLearning(env, gamma)
QLearnOpponent = QLearning(env, gamma)
game1 = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, QLearnAgent, QLearnOpponent)
QLearnErr = game1.train()
plt.plot(QLearnErr, linewidth=0.5)
plt.show()

plt.plot(QLearnErr, linewidth=0.5)
plt.ylim(0, 0.01)

FoeQAgent = FoeQ(env, gamma)
FoeQOpponent = FoeQ(env, gamma)
game2 = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, FoeQAgent, FoeQOpponent)
FoeQErr = game2.train()
plt.plot(FoeQErr, linewidth=0.5)
plt.show()

plt.plot(FoeQErr, linewidth=0.5)
plt.ylim(0, 0.01)

FriendQAgent = FriendQ(env, gamma)
FriendQOpponent = FriendQ(env, gamma)
game3 = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, FriendQAgent, FriendQOpponent)
FriendQErr = game3.train()
plt.plot(FriendQErr, linewidth=0.5)
plt.show()

plt.plot(FriendQErr, linewidth=0.5)
plt.ylim(0, 0.01)

CEQAgent = CEQ(env, gamma)
CEQOpponent = CEQ(env, gamma)
game4 = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, CEQAgent, CEQOpponent)
CEQErr = game4.train()
plt.plot(CEQErr, linewidth=0.5)
plt.show()

plt.plot(CEQErr, linewidth=0.5)
plt.ylim(0, 0.01)

CEQvsFoe = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, CEQAgent, FoeQOpponent)
print(CEQvsFoe.evaluate())

CEQvsFriend = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, CEQAgent, FriendQOpponent)
print(CEQvsFriend.evaluate())

FoevsFriend = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, FoeQAgent, FriendQOpponent)
print(FoevsFriend.evaluate())

FriendvsFoe = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, FriendQAgent, FoeQOpponent)
print(FriendvsFoe.evaluate())

CEQvsQLearn = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, CEQAgent, QLearnOpponent)
print(CEQvsQLearn.evaluate())

FoeQvsQLearn = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, FoeQAgent, QLearnOpponent)
print(FoeQvsQLearn.evaluate())

FriendQvsQLearn = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, FriendQAgent, QLearnOpponent)
print(FriendQvsQLearn.evaluate())

randomAgent= randomPlayAgent(env, gamma)
CEQvsRandom = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, CEQAgent, randomAgent)
print("CEQ vs Random: ", CEQvsRandom.evaluate())
FoeQvsRandom = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, FoeQAgent, randomAgent)
print("FoeQ vs Random: ", FoeQvsRandom.evaluate())
FriendQvsRandom = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, FriendQAgent, randomAgent)
print("FriendQ vs Random: ", FriendQvsRandom.evaluate())
QLearnvsRandom = SoccerGame(numEpisode, alpha_start, alpha_decay, alpha_min, epsilon_start, epsilon_decay, epsilon_min, gamma, env, QLearnAgent, randomAgent)
print("QLearn vs Random: ", QLearnvsRandom.evaluate())