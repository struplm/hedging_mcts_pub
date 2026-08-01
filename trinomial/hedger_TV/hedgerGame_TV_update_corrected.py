from __future__ import print_function
import sys

sys.path.append("..")
import numpy as np
from random import randrange
from scipy.stats import norm

##################################################################################
######MI CONFIGURATION#########################################################
##################################################################################


class HedgerPlan_TV:
    def __init__(self, config):
        self.config = config
        self.episode_step = 0
        # GBM Parameters
        self.n_actions = 20
        self.n_samples = 5  # number of individual actions per episode
        self.n_episodes = config[
            "reservoir"
        ]  # this is used a repository of possible paths
        self.measureSampleEfficiency = config["measureSampleEfficiency"]
        self.T = 1
        self.dt = self.T * 1 / self.n_samples
        self.s0 = 5
        # self.sigma = 0.3
        self.strike = 5
        self.transactionCosts = 1.0
        self.lambd = 1
        self.initPrice = 5
        # elements of the trinomial model
        # self.upFactor = np.exp(self.sigma*np.sqrt(2*self.dt))
        # self.downFactor = 1/self.upFactor

        # tmpUp = np.exp(self.sigma*np.sqrt(0.5*self.dt))
        # self.tmpUp = tmpUp
        self.probUp = 0.2
        self.probDown = 0.2
        self.probMid = 1 - self.probUp - self.probDown
        self.ttTransitions = self.createData()
        self.cumSumTransitions = np.cumsum(self.ttTransitions, axis=1).astype(int)
        print(
            "Terminal Variance Hedging with transaction cost: Updated version "
            + str(self.cumSumTransitions.shape)
        )

    def _euro_vanilla_delta(self, S, K, T, r, sigma):
        if T == 0.0:
            delta = 0.0
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1, 0.0, 1.0)
        return delta

    def conditionPrices(self):
        condPrices = np.zeros((self.n_episodes, self.n_samples + 1))
        for i in range(self.n_episodes):
            for j in range(self.n_samples + 1):
                if j == 0:
                    condPrices[i, 0] = self.pricesBS[0, 0]
                else:
                    condPrices[i, j] = self.pricesBS[
                        (-1) * self.cumSumTransitions[i, j] + j, j
                    ]
        return condPrices

    def createBSPrices(self):
        prices = np.zeros((2 * self.n_samples + 2, self.n_samples + 1))
        for i in range(2 * self.n_samples - 1):
            prices[i, self.n_samples] = max(
                0.0,
                self.s0 * pow(self.upFactor, self.n_samples) * pow(self.downFactor, i)
                - self.strike,
            )

        for j in range(self.n_samples):
            for i in range(2 * self.n_samples - 2 * (j) - 1):
                prices[i, self.n_samples - (j + 1)] = (
                    self.probDown * prices[i + 2, self.n_samples - j]
                    + self.probMid * prices[i + 1, self.n_samples - j]
                    + self.probUp * prices[i, self.n_samples - j]
                )
        return prices

    def createData(self):
        ttTransitions = np.zeros((self.n_episodes, self.n_samples + 1))
        for it in range(self.n_episodes):
            tmp = np.random.choice(
                [1, 0, -1], self.n_samples, p=[self.probUp, self.probMid, self.probDown]
            )
            ttTransitions[it, 1 : self.n_samples + 1] = tmp
        return ttTransitions

    def getEuroVanillaDelta(self, state):
        return self._euro_vanilla_delta(
            state[0, 3], self.strike, state[0, 4] / 365, 0, self.sigma
        )

    def getInitState(self):
        q = randrange(self.n_episodes)
        # state 0: counter
        # state 1: number of shares held
        # state 2: riskfree account Balance
        # state 3: new stockPrice
        # state 4: time to maturity
        state = (
            0,
            0.4,
            0.0,  # bank account! This is part of the optimization problem. IT SHOULD BE EITHER be optimized over, or chosen right a priori.
            self.initPrice,
            self.T,
        )
        cost = np.zeros(5) + 0.000001
        stateLog = (
            np.log(state[3]),
            np.log(state[3]),
            np.log(state[3]),
            np.log(state[3]),
            np.log(state[3]),
        )
        return np.array([state, stateLog, cost, state, state])

    def getSize(self):
        return (5, 5)

    def getActionSize(self):
        return self.n_actions

    def getNextState(self, state, dummy, action):
        newHoldings = action / self.n_actions
        # tmp = self.transactionCosts * abs(state[0,1] + newHoldings)
        c_0 = 0.05 / 0.2
        c_1s = 0.05
        tmp = np.minimum(c_0 * abs(state[0, 1] - newHoldings), c_1s)
        # new state. State corresponding to t is updated just before t+1.
        # state 0: counter
        # state 1: number of shares held
        # state 2: riskfree account Balance
        # state 3: new stockPrice
        # state 4: time to maturity
        if self.measureSampleEfficiency:
            marketTransition = 0
        else:
            if state[0, 3] > 1 and state[0, 3] < 9:
                marketTransition = (
                    state[0, 3]
                    + np.random.choice(
                        [1, 0, -1], 1, p=[self.probUp, self.probMid, self.probDown]
                    )[0]
                )
            if state[0, 3] == 1:
                marketTransition = (
                    state[0, 3]
                    + np.random.choice(
                        [1, 0], 1, p=[self.probUp, self.probMid + self.probDown]
                    )[0]
                )
            if state[0, 3] == 9:
                marketTransition = (
                    state[0, 3]
                    + np.random.choice(
                        [0, -1], 1, p=[self.probUp + self.probMid, self.probDown]
                    )[0]
                )

        stateNew = (
            state[0, 0] + 1,
            newHoldings,
            state[0, 2] - state[0, 3] * (-state[0, 1] + newHoldings) - tmp,
            marketTransition,
            state[0, 4] - self.dt,
        )

        cost = (
            state[2, 2] + tmp,
            state[2, 2] + tmp,
            state[2, 2] + tmp,
            state[2, 2] + tmp,
            state[2, 2] + tmp,
        )
        stateNewLog = (
            np.log(marketTransition),
            np.log(marketTransition),
            np.log(marketTransition),
            np.log(marketTransition),
            np.log(marketTransition),
        )

        return (np.array([stateNew, stateNewLog, cost, stateNewLog, stateNew]), dummy)

    def stateValue(self, state):
        return state[0, 2] + state[0, 3] * state[0, 1]
        # P_{t+1} = cashNew + S_{t+1}*newholding = cashOld - S_t*actOld + S_{t+1}*(actOld + holdingOld)
        # = cashOld + S_t*holdingOld + Delta S_{t+1} (actOld + holdingsOld)
        # = P_t + Delta S_{t+1} holdingsNew

    def getGameEnded(self, state, dummy):
        ts = state[0, :]
        if ts[0] < self.n_samples:
            return 0
        else:
            tmp = 100 * (
                1
                / self.s0**2
                * (self.stateValue(state) - max(0.0, state[0, :][3] - self.strike)) ** 2
            )  # +1/self.s0*2/self.lambd*state[2,2])
            # print('loss'+str((self.stateValue(state)-max(0.0,state[0,:][3]-self.strike))**2))
            # print('cost' + str(state[2,2]))
            if tmp > 20:  # 10
                tmp = 20  # 10
            # print(-tmp/5+1)
            return -tmp / 10 + 1  # /5

    def getSymmetries(self, state, pi):
        return [(state, pi)]

    def getNormalizations(self):
        stateNormalizations = (
            self.n_samples,
            self.n_samples,
            self.initPrice,
            self.s0,
            self.T,
        )
        if abs(self.transactionCosts) < 0.00001:  # dummy
            costNormalizations = (1, 1, 1, 1, 1)
        else:
            costNormalizations = (
                self.transactionCosts * self.s0 * self.n_samples,
                self.transactionCosts * self.s0 * self.n_samples,
                self.transactionCosts * self.s0 * self.n_samples,
                self.transactionCosts * self.s0 * self.n_samples,
                self.transactionCosts * self.s0 * self.n_samples,
            )
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
