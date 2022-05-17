from agents import ISoccerGameAgent
import numpy as np
from cvxopt import solvers, matrix
import copy


class FoeQ(ISoccerGameAgent):
    """The FoeQ (aka minimax Q-learning) agent.

    Reference:

    Michael L. Littman. 1994. "Markov games as a framework for multi-agent
    reinforcement learning". In Proceedings of the Eleventh International
    Conference on International Conference on Machine Learning (ICML '94).
    Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 157–163."""

    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        stateSpace = env.state_space
        actSpace = env.action_space
        dimOfQ = np.concatenate((stateSpace, [actSpace, actSpace]))
        dimOfPi = np.concatenate((stateSpace, [actSpace]))
        self.Q = np.ones(dimOfQ)
        self.V = np.ones(stateSpace)
        self.pi = np.ones(dimOfPi) / actSpace
        solvers.options["show_progress"] = False

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

    # See Greenwald, Hall, and Zinkevich, 2005, table 2
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

        # Step 3: update the policy.
        #
        # Given the game matrix in the current state
        #
        #               A
        #     N   S   E   W   stay
        #   N                       V
        #   S                       V
        #   E                       V
        # B W                       V
        #   stay                    V
        #
        # The values at each position is Q(s, A, B).
        #
        # The game matrix is Q(s, A, B) transposed.
        #
        # We need to solve pi(N), pi(S), pi(E), pi(W), pi(stay) using LP
        # from HW6
        #
        # Given the constraints:
        #
        # pie(rock) pie(paper) pie(scissor)   V
        #                1        -1         -1     >= 0
        #    -1                    1         -1     >= 0
        #     1         -1                   -1     >= 0
        #     1          1         1          0     >= 1
        #    -1         -1        -1          0     >=-1
        #
        # maximize
        #     0          0         0          1
        #
        # The matrix changes according to the rewards.
        #
        # Here, it is done using the same setup, but the numbers come from the
        # Q table.

        # First form game matrix
        gameMatrix = copy.copy(self.Q[s0, s1, s2].T)

        # Add constrains.
        #
        # All probabilities >= 0.
        numActions = self.env.action_space
        I = np.zeros((numActions, numActions))
        for i in range(numActions):
            I[i, i] = 1

        gameMatrix = np.vstack((gameMatrix, I))

        # Add V to the matrix.
        gameMatrix = np.hstack(
            (
                gameMatrix,
                [[-1], [-1], [-1], [-1], [-1], [0], [0], [0], [0], [0]],
            )
        )

        # Add constraints that all probabilities sum to 1.
        gameMatrix = np.vstack(
            (gameMatrix, [1, 1, 1, 1, 1, 0], [-1, -1, -1, -1, -1, 0])
        )

        gameMatrix = matrix(gameMatrix)
        b = matrix(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
        )
        c = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        # Use negative sign to convert max to min.
        sol = solvers.lp(-c, -gameMatrix, -b)

        result = np.array(sol["x"])

        # Update pi.
        self.pi[s0, s1, s2] = copy.copy(result[0:5, 0])

        # Step 4a: update V.
        # Here, V is the minimax value calculated by LP, so no need to
        # re-calculate it.
        # Note that only V is updated, but not V', because Q' and pi' are not
        # changed, and we are saving all V values.
        self.V[s0, s1, s2] = result[5, 0]

        # Step 4b: update Q value (on policy).
        # Q[s, a, o] = (1 - alpha) * Q[s, a, o] + alpha *
        # ((1 - gamma) * rew + gamma * V[s'])
        if not done:
            self.Q[s0, s1, s2, action, opponentAction] = (1 - alpha) * self.Q[
                s0, s1, s2, action, opponentAction
            ] + alpha * (
                (1 - self.gamma) * reward
                + self.gamma * self.V[s_prime0, s_prime1, s_prime2]
            )
        else:
            self.Q[s0, s1, s2, action, opponentAction] = (1 - alpha) * self.Q[
                s0, s1, s2, action, opponentAction
            ] + alpha * (1 - self.gamma) * reward
