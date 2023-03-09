import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = [[0 for _ in range(n_actions)] for _ in range(n_states)]
        self.gamma = 0.
        
    def select_action(self, state):
        e_greedy = np.zeros(self.n_actions)
        e_greedy.fill(self.epsilon / (self.n_actions - 1))
        e_greedy[np.argmax(self.Q[state])] = 1 - self.epsilon
        return np.random.choice(self.n_actions, p = e_greedy)
        
    def update(self, state, action, reward):
        self.Q[state][action] = reward

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass