import numpy as np
import random

# Environment class
class SocialLearningEnv:
    def __init__(self, num_actions, num_preferences):
        self.num_actions = num_actions
        self.num_preferences = num_preferences
        self.state = None
        self.preference = np.random.choice(self.num_preferences)  # Random initial preference

    def reset(self):
        self.state = np.random.choice(self.num_preferences)  # Randomly initialize the state
        return self.state

    def step(self, action):
        correct_recommendation = action == self.state  # Reward is 1 if recommendation is correct
        reward = 1 if correct_recommendation else -1

        next_state = np.random.choice(self.num_preferences)  # Randomly transition to a new state
        done = False  # No terminal state in this simple scenario

        print(f"Action taken: {self.preference}, Current State: {self.state}, Reward: {reward}")

        return next_state, reward, done, {}

    #def render(self):
        #print(f"REPRINT: Action taken (Reco): {self.preference}, Current State (Pref): {self.state}")


# Agent class
class QLearningAgent:
    def __init__(self, num_preferences, action_space, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.99):
        self.num_preferences = num_preferences
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((self.num_preferences, self.action_space))

    def decay_epsilon(self):
        # Decay epsilon to encourage more exploitation over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)  # Random exploration
        else:
            return np.argmax(self.q_table[state])  # Exploit best action

    def learn(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])  # Best action for the next state
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action])

    def get_q_table(self):
        return self.q_table
