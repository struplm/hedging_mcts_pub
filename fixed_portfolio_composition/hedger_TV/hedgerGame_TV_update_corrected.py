from __future__ import print_function
import sys

sys.path.append("..")
import numpy as np
from random import randrange
from scipy.stats import norm


class HedgerPlan_TV:
    def __init__(self, config):
        self.config = config
        self.episode_step = 0
        # GBM Parameters
        self.n_actions = 21
        self.n_samples = 5  # dont change
        self.n_episodes = config[
            "reservoir"
        ]  # this is used a repository of possible paths
        # self.ttPaths = np.random.randn(self.n_episodes,self.n_samples).astype(np.double)
        self.T = self.n_samples / 365
        self.dt = self.T * 1 / self.n_samples

    def getInitState(self):
        # state 0: counter
        # state 1: number of shares/last action
        # state 2: ACCUMULATED REWARD, warning it starts with one, for the log to be finite
        # state 3: new stockPrice
        # state 4: time to maturity
        state = (
            0,
            int(0),
            int(1),  # dummy
            int(np.random.randn(1).astype(np.double)[0] > 0),  # init market state
            self.T * 365,
        )

        cost = np.zeros(5) + 0.000001
        return np.array([state, state, cost, state, state])

    def getSize(self):
        return (5, 5)

    def getActionSize(self):
        return self.n_actions

    def get_rewards(self, actions, right_decision):
        modeval = np.array([200.0, 100.0])
        epss = modeval ** (-1)
        modearg = np.zeros([1, 2])
        modearg[:, 0] = right_decision
        modearg[:, 1] = 1 - right_decision
        modearg = modearg - 0.5
        reward = (((actions - modearg) ** 2 + epss[None, :]) ** (-1)).sum()
        return reward

    def getNextState(self, state, dummy, action):
        newHoldings = (action - (self.n_actions - 1) / 2) * 2 / (self.n_actions - 1)
        reward = self.get_rewards(newHoldings, int(state[0, 3]))
        marketTransition = int(np.random.randn(1).astype(np.double)[0] > 0)

        stateNew = (
            state[0, 0] + 1,
            newHoldings,
            state[0, 2] + reward,
            marketTransition,
            state[0, 4] - self.dt * 365,
        )

        return (
            np.array([stateNew, stateNew, np.zeros(5) + 0.000001, stateNew, stateNew]),
            dummy,
        )

    def getGameEnded(self, state, dummy):
        ts = state[0, :]
        if ts[0] < self.n_samples:
            return 0
        else:
            tmp = 1 / 1000 * (state[0, 2])
            return tmp

    def getSymmetries(self, state, pi):
        return [(state, pi)]

    def getNormalizations(self):
        stateNormalizations = (self.n_samples, self.n_samples, 1.0, 1.0, self.T * 365)
        costNormalizations = (1, 1, 1, 1, 1)
        return np.array(
            [
                stateNormalizations,
                stateNormalizations,
                costNormalizations,
                stateNormalizations,
                stateNormalizations,
            ]
        )

    def stringRepresentation(self, state):
        return (
            str(state[0, 0])
            + "_"
            + str(round(state[0, 1], 2))
            + "_"
            + str(round(state[0, 2], 2))
            + "_"
            + str(round(state[0, 3], 2))
            + "_"
            + str(round(state[2, 2], 2))
        )
