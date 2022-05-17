from agents import ISoccerGameAgent
from cvxopt import solvers, matrix
import numpy as np


class CEQ(ISoccerGameAgent):
    """The Correlated Q-Learning agent.

    This agent is similar to FoeQ, but differs in the calculation of pi.

    Reference:

    Amy Greenwald, Keith Hall and Martin Zinkevich. 2005.
    "Correlated Q-Learning". Brown University Technical Report. CS-05-08."""

    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        stateSpace = env.state_space
        actSpace = env.action_space
        dimOfQ = np.concatenate((stateSpace, [actSpace, actSpace]))
        self.Q = np.ones(dimOfQ)
        self.V = np.ones(stateSpace)

        # This is different from FoeQ.
        # calculate probability for each agent-opponent action pair.
        self.pi = np.ones(dimOfQ) / (actSpace**2)

        # This is also different from FoeQ because CEQ is a joint distribution,
        # we need to simulate the opponent's utilities too.
        self.opponentQ = np.ones(dimOfQ)
        self.opponentV = np.ones(stateSpace)
        solvers.options["show_progress"] = False

    def act(self, s0, s1, s2):
        s2 = int(s2)
        actProb = np.sum(self.pi[s0, s1, s2, :], axis=1)
        rand = np.random.random()
        prob = 0
        for i in range(len(actProb)):
            prob += actProb[i]
            if rand < prob:
                return i
        return i

    # See Greenwald, Hall, and Zinkevich, 2005, table 2.
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

        # Step 3: update policy.
        #
        # Given the game matrix in the current state
        #               A
        #     N   S   E   W   stay
        #   N
        #   S     QA(s,A,B), QB(s,A,B), pi(s,A,B)
        # B E
        #   W
        #   stay
        #
        # The values at each position for A is Q(s, A, B), for B is Qo(s, A, B)
        # (Opponent)
        #
        # The joint probability is pi(s, A, B)
        #
        # The constraints of the Linear Programming (LP) problem are:
        #
        # Set 1:
        #
        # For the Correlated Equilibrium to hold, no agent should change
        # action given the CE policy.
        #
        # For example, if A chooses N, changing it won't increase its value.
        #
        # pi(N, N)Q(s, N, N) +... + pi(N, stay)Q(s, N, stay) >=
        # pi(N, N)Q(s, S, N) + ... + pi(N, stay)Q(s, S, stay)
        # [were A to switch from N to S]
        #
        # ...
        #
        # pi(N, N)Q(s, N, N)+ ... + pi(N, stay)Q(s, N, stay) >=
        # pi(N, N)Q(s, stay, N) + ... + pi(N, stay)Q(s, stay, stay)
        # [were A to switch from N to stay]
        #
        # This is a set of 4 equations (from N to 4 alternatives).
        #
        # There are 5 sets for A (20 total)
        #
        # Same for B, notice we are using Qo here
        #
        # pi(N, N)Qo(s, N, N) + ... + pi(stay, N)Qo(s, stay, N) >=
        # pi(N, N)Qo(s, N, S) + ... + pi(stay, N)Qo(s, stay, N)
        # [were B to switch from N to S]
        #
        # ...
        #
        # pi(N, N)Qo(s, N, N) + ... + pi(stay, N)Qo(s, stay, N) >=
        # pi(N, N)Qo(s, N, stay) + ... + pi(stay, N)Qo(s, stay, stay)
        # [were B to switch from N to stay]
        #
        # Together they define the CE policy.

        # Set 2:
        #
        # Constraints for the probabilities (>=0 and add to 1)

        # Set 3:
        #
        # In addition, we need another criterium to pick which CE
        #
        # See Greenwald, Hall, and Zinkevich, 2005, equation 17-20.
        #
        # For example, utilitarian: maximize the sum of all agents' rewards:
        # at state s maximize sum[pi(actionA, actionB)Q(s, A, B)]
        #
        # Altogether, we can solve the LP and get policy for both A and B

        # The final LP problem has the format
        #
        # pi(N, N) + ... + pi(N, stay) + pi(S, N) + ... + pi(S, stay) + ... +
        # pi(stay, N) + ... + pi(stay, stay)
        #
        # or
        #
        # pi(s, 0, 1) + ... + pi(s, 0, 4) + pi(s, 1, 1) + ... + pi(s, 1, 4)
        # + ... + pi(s, 4, 1) + ... + pi(s, 4, 4).
        #
        # This is the column order for LP.
        #
        # The index of pi(s, m, n) is therefore 5 * m + n. See below.

        # Construct linear functions
        #
        # Set 1
        numActions = self.env.action_space
        numComActions = numActions**2  # Number of combined actions.
        A = []
        b = []

        # First do A
        for i in range(numActions):
            for j in range(numActions):
                if i == j:
                    continue
                equation = [0] * (numComActions)
                for k in range(numActions):
                    # Given action i, no need for A to change to action j.
                    # For all k, sum(pi(s, i, k)[Q(s, i, k) - Q(s, j, k)]) >= 0
                    # Locate the column position in the LP matrix A.
                    index = numActions * i + k
                    equation[index] = (
                        self.Q[s0, s1, s2, i, k] - self.Q[s0, s1, s2, j, k]
                    )
                A.append(equation)
                b.append(0)

        # Next do B
        for i in range(numActions):
            for j in range(numActions):
                if i == j:
                    continue
                equation = [0] * (numComActions)
                for k in range(numActions):
                    # Given action i, no need for B to change to action j.
                    # For all k,
                    # sum(pi(s, k, i)[Qo(s, k, i) - Qo(s, k, j)]) >= 0
                    # Locate the column position in the LP matrix A.
                    index = numActions * k + i
                    equation[index] = (
                        self.opponentQ[s0, s1, s2, k, i]
                        - self.opponentQ[s0, s1, s2, k, j]
                    )
                A.append(equation)
                b.append(0)

        # Set 2
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        I = np.zeros((numComActions, numComActions))
        for i in range(numComActions):
            I[i, i] = 1
        A = np.vstack((A, I, [1] * numComActions, [-1] * numComActions))
        b = np.concatenate((b, [0] * numComActions, [1, -1]))

        # Set 3
        c = []
        for i in range(numActions):
            for j in range(numActions):
                c.append(
                    self.Q[s0, s1, s2, i, j] + self.opponentQ[s0, s1, s2, i, j]
                )

        # Solve the linear programming problem.
        A = matrix(A)
        b = matrix(b)
        c = matrix(c)
        sol = solvers.lp(-c, -A, -b)
        result = np.array(sol["x"])
        result = np.reshape(result, (numActions, numActions))

        # Update pi.
        self.pi[s0, s1, s2] = result

        # Step 4a: update V for both agent and opponent
        # Note that only V is updated, but not V', because Q' and pi' are not
        # changed and we are saving all V values.
        self.V[s0, s1, s2] = np.sum(self.pi[s0, s1, s2] * self.Q[s0, s1, s2])

        self.opponentV[s0, s1, s2] = np.sum(
            self.pi[s0, s1, s2] * self.opponentQ[s0, s1, s2]
        )

        # Step 4b: update Q value (on policy) for both agent and opponent.
        # Q[s, a, o] = (1 - alpha) * Q[s, a, o] + alpha *
        # ((1 - gamma) * rew + gamma * V[s'])
        if not done:
            self.Q[s0, s1, s2, action, opponentAction] = (1 - alpha) * self.Q[
                s0, s1, s2, action, opponentAction
            ] + alpha * (
                (1 - self.gamma) * reward
                + self.gamma * self.V[s_prime0, s_prime1, s_prime2]
            )
            self.opponentQ[s0, s1, s2, action, opponentAction] = (
                1 - alpha
            ) * self.opponentQ[s0, s1, s2, action, opponentAction] + alpha * (
                (1 - self.gamma) * opponent_reward
                + self.gamma * self.opponentV[s_prime0, s_prime1, s_prime2]
            )
        else:
            self.Q[s0, s1, s2, action, opponentAction] = (1 - alpha) * self.Q[
                s0, s1, s2, action, opponentAction
            ] + alpha * (1 - self.gamma) * reward
            self.opponentQ[s0, s1, s2, action, opponentAction] = (
                1 - alpha
            ) * self.opponentQ[s0, s1, s2, action, opponentAction] + alpha * (
                1 - self.gamma
            ) * opponent_reward
        pass
