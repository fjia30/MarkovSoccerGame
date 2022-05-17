# Soccer class resembles an enviroment in OpenAI Gym
# Use reset() to initiate an episode
# Use step(actionA, actionB) to simulate an action which returns next stata, reward and isFinished
# Use render() to draw the current state
# self.action_space: num of actions
# self.state_space: <num of variabel1, num of variable2, num of variable3>

# the fild is a 2x4 grid
# number the grid as
# 0, 1, 2, 3
# 4, 5, 6, 7
# states are position of A, position of B and whether A or B has the ball
# actions for both A and B are (N,S,E,W,stick) which is represented as 0~4
import numpy as np

GOAL_REWARD = 100


class SoccerEnviroment:
    def __init__(self):
        self.actions = [-4, 4, 1, -1, 0]
        self.action_space = len(self.actions)
        self.state_space = (8, 8, 2)

    def __showCurrentState(self):
        return (self.posOfA, self.posOfB, self.AHasBall)

    # returns the reward for A, the reward for B is the negative by definition of zero sum game
    def __calculateReward(self):
        if self.AHasBall:
            if self.posOfA == 0 or self.posOfA == 4:
                return GOAL_REWARD
            if self.posOfA == 3 or self.posOfA == 7:
                return -GOAL_REWARD
        else:
            if self.posOfB == 0 or self.posOfB == 4:
                return GOAL_REWARD
            if self.posOfB == 3 or self.posOfB == 7:
                return -GOAL_REWARD
        return 0

    # calculate the postion of a player after a move
    # player sticks if moving towards a wall
    def __movePlayer(self, postion, action):
        newPostion = postion + self.actions[action]
        if newPostion < 0 or newPostion > 7:
            return postion
        else:
            return newPostion

    def __moveA(self, actionOfA):
        newPosOfA = self.__movePlayer(self.posOfA, actionOfA)
        if newPosOfA != self.posOfB:
            self.posOfA = newPosOfA
        # if A run into B with a ball, give the ball to B
        elif self.AHasBall:
            self.AHasBall = False

    def __moveB(self, actionOfB):
        newPosOfB = self.__movePlayer(self.posOfB, actionOfB)
        if newPosOfB != self.posOfA:
            self.posOfB = newPosOfB
        # if B run into A with a ball, give the ball to A
        elif not self.AHasBall:
            self.AHasBall = True

    # initilized game with random ball poccession
    def reset(self):
        self.posOfA, self.posOfB = np.random.choice([1, 2, 5, 6], size=2, replace=False)
        self.AHasBall = np.random.choice([True, False])
        return self.__showCurrentState()

    # take a step in the game given actions of A and B
    # return next state, reward and whether the game is dones
    def step(self, actionOfA, actionOfB):
        if np.random.random() > 0.5:
            # A moves first
            self.__moveA(actionOfA)
            self.__moveB(actionOfB)
        else:
            # B moves first
            self.__moveB(actionOfB)
            self.__moveA(actionOfA)

        reward = self.__calculateReward()
        return self.__showCurrentState(), reward, not reward == 0

    def render(self):
        out = "---------------------\n"
        for i in range(2):
            for j in range(4):
                position = i * 4 + j
                if self.posOfA == position:
                    if self.AHasBall:
                        out += "| A* "
                    else:
                        out += "| A  "
                elif self.posOfB == position:
                    if not self.AHasBall:
                        out += "| B* "
                    else:
                        out += "| B  "
                else:
                    out += "|    "
            out += "|\n---------------------\n"
        print(out)
