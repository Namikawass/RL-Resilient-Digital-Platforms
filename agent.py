"""
agent.py
--------
Neural Computation Unit part of the project.

A minimal Q-learning agent (tabular, epsilon-greedy). Kept
deliberately simple and dependency-light (numpy only) so the
learning dynamics -- state, action, reward, policy update -- are
fully transparent. This is the natural first step before moving to
a Deep Q-Network or actor-critic version with PyTorch.
"""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.15, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.98):
        self.n_actions = n_actions
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        # Q-table: maps state tuple -> array of action values
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
