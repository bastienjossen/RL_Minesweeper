import random
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.n_actions = n_actions
        self.lr        = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        # Q-table: state_key → vector of length n_actions
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

    def state_key(self, state: np.ndarray):
        # quantize floats into small ints, then flatten to tuple
        # (two channels, values [0,1] mapped to 0–255)
        flat = (state * 255).astype(np.int16).flatten()
        return tuple(flat.tolist())

    def act(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        flat_mask  = legal_mask.flatten()
        legal_idxs = np.nonzero(flat_mask)[0]
        if len(legal_idxs) == 0:
            # no legal moves (shouldn’t happen) → pick random
            return random.randrange(self.n_actions)

        # ε-greedy over legal
        if random.random() < self.epsilon:
            return int(np.random.choice(legal_idxs))

        # exploitation: pick best Q among legal
        key   = self.state_key(state)
        qvals = self.Q[key]
        legal_q = qvals[legal_idxs]
        best = np.flatnonzero(legal_q == legal_q.max())
        return int(np.random.choice(legal_idxs[best]))

    def update(self, state, action, reward, next_state):
        k  = self.state_key(state)
        k2 = self.state_key(next_state)
        target = reward + self.gamma * np.max(self.Q[k2])
        self.Q[k][action] += self.lr * (target - self.Q[k][action])

    def reset(self):
        self.Q.clear()
