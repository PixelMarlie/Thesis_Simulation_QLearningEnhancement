import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Environment class
class SocialLearningEnv:
    def __init__(self, num_actions):
        # Define the size of the feature vectors
        self.num_user_features = 5  # Example: age, interests, history
        self.num_session_features = 3  # Example: time of day, device
        self.num_feedback_features = 2  # Example: likes, skips

        self.num_features = self.num_user_features + self.num_session_features + self.num_feedback_features
        self.num_actions = num_actions

        # Simulated user preferences and contexts
        self.user_profile = np.random.rand(self.num_user_features)  # Randomly generated profile
        self.state = None
        self.episode_count = 0  # Track the number of episodes for penalty decay

    def reset(self):
        # Increment episode count
        self.episode_count += 1

        # Generate a new session context and feedback state
        session_context = np.random.rand(self.num_session_features)
        interaction_feedback = np.random.rand(self.num_feedback_features)

        self.state = np.concatenate([self.user_profile, session_context, interaction_feedback])
        return self.state

    def step(self, action):
        # Reward calculation based on engagement, novelty, and satisfaction
        engagement_score = np.random.rand()  # Simulate engagement score
        novelty_score = 1.0 if np.random.rand() > 0.5 else -1.0  # Example: random novelty
        satisfaction_score = np.random.choice([0, 1])  # Example: binary satisfaction

        reward = (0.5 * engagement_score) + (0.3 * novelty_score) + (0.2 * satisfaction_score)

        # Add a penalty for incorrect actions (simulating mistakes)
        penalty_factor = max(0.1, 1.0 - (self.episode_count * 0.001))  # Reduce penalty over time
        penalty = penalty_factor * (1 - satisfaction_score)  # Higher penalty for unsatisfied actions
        reward -= penalty

        # Generate the next state (simulate user dynamics)
        session_context = np.random.rand(self.num_session_features)
        interaction_feedback = np.random.rand(self.num_feedback_features)
        next_state = np.concatenate([self.user_profile, session_context, interaction_feedback])

        done = False  # No terminal state in this example
        return next_state, reward, done, {}

class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QLearningAgent:
    def __init__(self, input_dim, num_actions, lr=0.001, gamma=0.9, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.99):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Neural network and optimizer
        self.q_network = QNetwork(input_dim, num_actions)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:  # Exploration
            return np.random.randint(self.num_actions)
        else:  # Exploitation
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        # Forward pass to get Q-values
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        # Compute the target Q-value
        target = reward_tensor + self.gamma * torch.max(next_q_values).item()

        # Compute loss and update the network
        loss = self.loss_fn(q_values[0, action], target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
