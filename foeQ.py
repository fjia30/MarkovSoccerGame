from agents import *

# FoeQ / minimaxQ is implemented according to Littman 1994
class FoeQ(SoccerGameAgent):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        stateSpace = env.state_space
        actSpace = env.action_space
        dimOfQ = np.concatenate((stateSpace, [actSpace, actSpace]))
        dimOfPi = np.concatenate((stateSpace, [actSpace]))
        self.Q = np.ones(dimOfQ)
        self.V = np.ones(stateSpace)
        self.pi = np.ones(dimOfPi) / actSpace
        solvers.options['show_progress'] = False

    def act(self, s0, s1, s2):
        s2 = int(s2)
        actProb = self.pi[s0, s1, s2, :]
        rand = np.random.random()
        prob = 0
        for i in range(len(actProb)):
            prob += actProb[i]
            if rand < prob:
                return i
        return i

    # see Greenwald, Hall, and Zinkevich 2005 table 2
    def learn(self, alpha, s0, s1, s2, action, opponentAction, s_prime0, s_prime1, s_prime2, reward, opponent_reward, done):
        s2 = int(s2)
        s_prime2 = int(s_prime2)

        # step 3. update policy
        # given game matrix in current state
        #
        #               A
        #     N   S   E   W   stay
        #   N                       V
        #   S                       V
        #   E                       V
        # B W                       V
        #   stay                    V
        #
        # the values at each position is Q(s, A, B)
        # game matix is Q(s,A,B) transposed
        # we need to solve pi(N), pi(S), pi(E), pi(W), pi(stay) using LP
        # from HW6, we have
            # given constraints
            # pie(rock) pie(paper) pie(sissor)    V
            #                1        -1         -1     >= 0
            #    -1                    1         -1     >= 0
            #     1         -1                   -1     >= 0
            #     1          1         1          0     >= 1
            #    -1         -1        -1          0     >=-1
            # maximize
            #     0          0         0          1
            # matrix changes according to the rewardss
        # here it is done using the same setup but the numbers come from the Q table

        # first form game matrix       
        gameMatrix = copy.copy(self.Q[s0, s1, s2].T)
        # add constrains all probability >= 0
        numActions = self.env.action_space
        I = np.zeros((numActions, numActions))
        for i in range(numActions):
            I[i, i] = 1
        gameMatrix = np.vstack((gameMatrix, I))
        # add V to the matrix
        gameMatrix = np.hstack(
            (gameMatrix, [[-1], [-1], [-1], [-1], [-1], [0], [0], [0], [0], [0]]))
        # add contraints that all probs sum to 1
        gameMatrix = np.vstack(
            (gameMatrix, [1, 1, 1, 1, 1, 0], [-1, -1, -1, -1, -1, 0]))
        gameMatrix = matrix(gameMatrix)
        b = matrix([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -1.])
        c = matrix([0., 0., 0., 0., 0., 1.])
        # use negative sign to convert max to min
        sol = solvers.lp(-c, -gameMatrix, -b)
        result = np.array(sol['x'])
        # update pie
        self.pi[s0, s1, s2] = copy.copy(result[0:5, 0])

        # step 4a. update V
        # here V is the minimax value calculated by lp so no need to re-calculate
        # note that only V is updated but not V' because Q' and pi' are not changed
        # and we are saving all V values
        self.V[s0, s1, s2] = result[5,0]
    
        # step 4b. update Q value (on policy)
        # Q[s,a,o] = (1-alpha) * Q[s,a,o] + alpha * ((1-gamma)*rew + gamma * V[s’])
        if not done:
            self.Q[s0, s1, s2, action, opponentAction] = \
                (1 - alpha) * self.Q[s0, s1, s2, action, opponentAction] + \
                alpha * ((1-self.gamma) * reward + self.gamma *
                self.V[s_prime0, s_prime1, s_prime2])
        else:
            self.Q[s0, s1, s2, action, opponentAction] = \
                (1 - alpha) * self.Q[s0, s1, s2, action,
                opponentAction] + alpha * (1 - self.gamma) * reward
        pass