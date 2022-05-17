from agents import ISoccerGameAgent
import numpy as np

# based on littman FFQ paper
# compared to FoeQ, we dont need to store pi and V because we can get both easily from the game matrix
# V is the max value from the matrix and pi chooses the action having that value
class FriendQ(ISoccerGameAgent):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        stateSpace = env.state_space
        actSpace = env.action_space
        dimOfQ = np.concatenate((stateSpace, [actSpace, actSpace]))
        self.Q = np.ones(dimOfQ)

    # create game matrix at state s
    #               A
    #     N   S   E   W   stay
    #   N
    #   S     Q(s, A, B)
    # B E
    #   W
    #   stay
    #
    # the values at each position is Q(s, A, B)
    # game matix is Q(s) transposed
    def __constructGameMatrix(self, s0, s1, s2: int):
        return self.Q[s0, s1, s2].T

    def act(self, s0, s1, s2):
        s2 = int(s2)
        # get game matrix at current state
        gameMatrix = self.__constructGameMatrix(s0, s1, s2)
        # get max value of each column
        columns = np.amax(gameMatrix, axis=0)
        # pick the best action, tie-break randomly
        # random tie-breaking is essensial because it is equal to dividing the probability among best options
        action = np.random.choice(np.flatnonzero(columns == columns.max()))
        return action

    # see Greenwald, Hall, and Zinkevich 2005 table 2
    def learn(
        self,
        alpha,
        s0,
        s1,
        s2,
        action,
        opponentAction,
        s_prime0,
        s_prime1,
        s_prime2,
        reward,
        opponent_reward,
        done,
    ):
        s2 = int(s2)
        s_prime2 = int(s_prime2)
        # step 4a. calculate V_prime which is max(B) max(A) in the game matrix littman FFQ paper equation 7
        # notice here we are making gameMatrix for next state s_prime
        # for end state, next state value is 0
        if not done:
            gameMatrix = self.__constructGameMatrix(s_prime0, s_prime1, s_prime2)
            V_prime = np.max(gameMatrix)
        else:
            V_prime = 0

        # step 4b. update Q, which is the same as in FoeQ
        # Q[s,a,o] = (1-alpha) * Q[s,a,o] + alpha * ((1-gamma)*rew + gamma * V[s’])
        # except that V_prime is already calcuated here
        self.Q[s0, s1, s2, action, opponentAction] = (1 - alpha) * self.Q[
            s0, s1, s2, action, opponentAction
        ] + alpha * ((1 - self.gamma) * reward + self.gamma * V_prime)
        pass
