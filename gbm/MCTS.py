import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    # Monte Carlo Tree Search

    def __init__(self, game, nnet, args, permanentNoise=0, firstLeafNoise=1):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.permanentNoise = permanentNoise
        self.firstLeafNoise = firstLeafNoise
        self.Es = {}
        self.first_leaf = True
        # Cache frequently used value
        self.nn_weight = args["nnWeight"]

    def getActionProb(self, state, temp=1):
        # Run all simulations
        for _ in range(self.args["numMCTSSims"]):
            self.search(state)

        self.first_leaf = True
        s = self.game.stringRepresentation(state)
        action_size = self.game.getActionSize()

        # Use list comprehension with get() method for efficiency
        counts = [self.Nsa.get((s, a), 0) for a in range(action_size)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # Apply temperature using numpy for performance
        if temp != 1:
            counts = np.power(counts, 1.0 / temp)
            counts_sum = float(np.sum(counts))
        else:
            counts_sum = float(sum(counts))

        if counts_sum == 0:
            print("AAAAAARGH")
            print("s: ")
            print(s)
            print(state)
            print(self.Nsa)
            print(self.Ps)
            print(self.Qsa)

        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state):
        s = self.game.stringRepresentation(state)

        # Check if game ended - more efficient with get()
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(state, 1)
        if self.Es[s] != 0:
            return self.Es[s]

        # Initialize new state
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(state)
            action_size = self.game.getActionSize()

            if self.first_leaf:
                self.first_leaf = False
                # Create noise only once per state
                noise = np.random.randint(
                    0, high=action_size, size=action_size, dtype=int
                )
                noise_sum = np.sum(noise)
                if noise_sum > 0:
                    noise = noise / noise_sum

                if self.firstLeafNoise == 1:
                    self.Ps[s] = noise
                else:
                    self.Ps[s] = (1 - self.firstLeafNoise) * self.Ps[
                        s
                    ] + self.firstLeafNoise * noise

            if self.permanentNoise:
                # Create permanent noise only once
                perm_noise = np.random.randint(
                    0, high=action_size, size=action_size, dtype=int
                )
                perm_noise_sum = np.sum(perm_noise)
                if perm_noise_sum > 0:
                    perm_noise = perm_noise / perm_noise_sum
                self.Ps[s] = (1 - self.permanentNoise) * self.Ps[
                    s
                ] + self.permanentNoise * perm_noise

            # Normalize probabilities efficiently
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] = self.Ps[s] / sum_Ps_s
            else:
                log.error("probs sum to zero")

            self.Ns[s] = 0
            return v

        # Find best action using UCB
        cur_best = -float("inf")
        best_act = -1

        # Pre-compute values used in the loop
        sqrt_ns = math.sqrt(self.Ns[s])
        action_size = self.game.getActionSize()

        # Remove unnecessary if True condition
        for a in range(action_size):
            sa_pair = (s, a)  # Cache tuple creation

            if sa_pair in self.Qsa:
                u = self.Qsa[sa_pair] + self.nn_weight * self.Ps[s][a] * sqrt_ns / (
                    1 + self.Nsa[sa_pair]
                )
            else:
                # Avoid redundant calculation with EPS
                u = self.nn_weight * self.Ps[s][a] * sqrt_ns

            if u > cur_best:
                cur_best = u
                best_act = a

        # Get next state
        a = best_act
        next_s, next_player = self.game.getNextState(state, 1, a)

        # Recursive search
        v = self.search(next_s)

        # Update statistics - cache the tuple
        sa_pair = (s, a)
        if sa_pair in self.Qsa:
            nsa = self.Nsa[sa_pair]  # Cache dictionary access
            self.Qsa[sa_pair] = (nsa * self.Qsa[sa_pair] + v) / (nsa + 1)
            self.Nsa[sa_pair] += 1
        else:
            self.Qsa[sa_pair] = v
            self.Nsa[sa_pair] = 1

        self.Ns[s] += 1
        return v


# import logging
# import math

# import numpy as np

# EPS = 1e-8

# log = logging.getLogger(__name__)


# class MCTS:
#     # Monte Carlo Tree Search

#     def __init__(self, game, nnet, args, permanentNoise=0, firstLeafNoise=1):
#         self.game = game
#         self.nnet = nnet
#         self.args = args
#         self.Qsa = {}
#         self.Nsa = {}
#         self.Ns = {}
#         self.Ps = {}
#         self.permanentNoise = permanentNoise
#         self.firstLeafNoise = firstLeafNoise
#         self.Es = {}
#         self.first_leaf = True
#         self.add_noise = True

#     def getActionProb(self, state, temp=1):
#         for i in range(self.args["numMCTSSims"]):
#             self.search(state)
#         self.first_leaf = True
#         s = self.game.stringRepresentation(state)
#         counts = [
#             self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
#             for a in range(self.game.getActionSize())
#         ]
#         if temp == 0:
#             bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
#             bestA = np.random.choice(bestAs)
#             probs = [0] * len(counts)
#             probs[bestA] = 1
#             return probs

#         counts = [x ** (1.0 / temp) for x in counts]
#         counts_sum = float(sum(counts))
#         if counts_sum == 0:
#             print("AAAAAARGH")
#             print("s: ")
#             print(s)
#             print(state)
#             print(self.Nsa)
#             print(self.Ps)
#             print(self.Qsa)
#         probs = [x / counts_sum for x in counts]
#         return probs

#     def search(self, state):
#         s = self.game.stringRepresentation(state)
#         if s not in self.Es:
#             self.Es[s] = self.game.getGameEnded(state, 1)
#         if self.Es[s] != 0:
#             return self.Es[s]

#         if s not in self.Ps:
#             self.Ps[s], v = self.nnet.predict(state)
#             if self.first_leaf:  # and self.add_noise:
#                 self.first_leaf = False
#                 noise = np.random.randint(
#                     0,
#                     high=self.game.getActionSize(),
#                     size=self.game.getActionSize(),
#                     dtype=int,
#                 )
#                 noise = noise / sum(noise)

#                 if self.firstLeafNoise == 1:
#                     self.Ps[s] = noise
#                 else:
#                     self.Ps[s] = (1 - self.firstLeafNoise) * self.Ps[
#                         s
#                     ] + self.firstLeafNoise * noise
#             if self.permanentNoise:
#                 self.Ps[s] = (1 - self.permanentNoise) * self.Ps[
#                     s
#                 ] + self.permanentNoise * np.random.randint(
#                     0,
#                     high=self.game.getActionSize(),
#                     size=self.game.getActionSize(),
#                     dtype=int,
#                 )
#             sum_Ps_s = np.sum(self.Ps[s])
#             if sum_Ps_s > 0:
#                 self.Ps[s] = self.Ps[s] / sum_Ps_s
#             else:
#                 log.error("probs sum to zero")
#             self.Ns[s] = 0
#             return v

#         cur_best = -float("inf")
#         best_act = -1

#         for a in range(self.game.getActionSize()):
#             if True:
#                 if (s, a) in self.Qsa:
#                     u = self.Qsa[(s, a)] + self.args["nnWeight"] * self.Ps[s][
#                         a
#                     ] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
#                 else:
#                     u = (
#                         self.args["nnWeight"]
#                         * self.Ps[s][a]
#                         * math.sqrt(self.Ns[s] + EPS)
#                     )

#                 if u > cur_best:
#                     cur_best = u
#                     best_act = a
#         a = best_act
#         next_s, next_player = self.game.getNextState(state, 1, a)

#         v = self.search(next_s)

#         if (s, a) in self.Qsa:
#             self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
#                 self.Nsa[(s, a)] + 1
#             )
#             self.Nsa[(s, a)] += 1

#         else:
#             self.Qsa[(s, a)] = v
#             self.Nsa[(s, a)] = 1

#         self.Ns[s] += 1
#         return v
