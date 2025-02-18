from QLearningSimulationENV import SocialLearningEnv, QLearningAgent
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Training setup
num_actions = 5  # Start small, then scale up
num_preferences = 5  # User preference complexity (state space)

# Allow for scaling (comment out to disable, include lines where "Plot nonzero Q-values over episodes (SOP 3)")
'''
scaling_factor = 2  # doubling it each test (e.g., 5 → 10 → 20 → 40)
num_actions *= scaling_factor
num_preferences *= scaling_factor
'''

env = SocialLearningEnv(num_actions, num_preferences)
agent = QLearningAgent(num_preferences=num_preferences, action_space=num_actions)

# Training parameters
num_episodes = 1000  # Reduce for quicker comparison
rewards = []
start_time = time.time()

# For visualizing table sparsity (SOP 3)
q_table = agent.get_q_table()

# For visualizing growth of q-table over time (SOP 3)
nonzero_q_values_history = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    states_in_episode = set()

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        # Learn and update Q-table
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        step_count += 1

        if step_count >= 1000:  # Limit max steps per episode
            done = True

    nonzero_q_values_history.append(np.count_nonzero(agent.get_q_table())) # Track nonzero Q-values
    rewards.append(total_reward)  # Track total reward for this episode
    agent.decay_epsilon()
    print(f"Episode {episode}: Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# Training complete
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Plot nonzero Q-values over episodes (SOP 3)
'''
action_sizes = [5, 10, 20, 40, 80]  # Increase exponentially
memory_usage = []
for actions in action_sizes:
    states = actions  # Assume num_preferences scales with num_actions
    env = SocialLearningEnv(actions, states)
    agent = QLearningAgent(num_preferences=states, action_space=actions)

    q_table_size_bytes = agent.get_q_table().nbytes
    memory_usage.append(q_table_size_bytes / (1024 * 1024))
# Plot memory usage growth
plt.figure(figsize=(10, 6))
plt.plot(action_sizes, memory_usage, marker='o', linestyle='-', color='red')
plt.xlabel("Number of Actions (Scaling Factor)")
plt.ylabel("Memory Usage (MB)")
plt.title("Q-Table Growth: Actions vs. Memory Usage")
plt.grid()
plt.savefig('q_table_memory_growth.png')
plt.show()
'''


# Q-Learning: dense state-action with no meaningful learning (SOP 3)
plt.figure(figsize=(12, 6))
sns.heatmap(q_table, cmap='coolwarm', annot=False)
plt.xlabel('Actions')
plt.ylabel('States')
plt.title('Q-Table Heatmap: State-Action Values')
plt.savefig('q_table_heatmap.png')  # Save plot
plt.show()

# Reward trend visualization (SOP 2)
plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Unenhanced Q-Learning Rewards', color='blue')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Unenhanced Q-Learning: Training Reward Trend')
plt.legend()
plt.savefig('unenhanced_reward_trend.png')  # Save plot as PNG
plt.show()

# Visualize Q-values for sampled states (SOP 1)
sampled_states = [env.reset() for _ in range(100)]
q_values = []
for state in sampled_states:
    state_tensor = np.array(state, dtype=np.float32).reshape(1, -1)  # Reshape for 2D array
    q_values.append(agent.get_q_table()[tuple(state_tensor.argmax(axis=1))])
q_values = np.array(q_values)  # Convert list to NumPy array
# Plot Q-values for each action
plt.figure(figsize=(10, 6))
for action in range(num_actions):
    plt.plot(q_values[:, action], label=f'Action {action}')
plt.xlabel('Sampled States')
plt.ylabel('Q-values')
plt.title('Unenhanced Q-Learning: Learned Q-Values for Sampled States')
plt.legend()
plt.savefig('unenhanced_q_values_plot.png')  # Save plot as PNG
plt.show()

# (SOP1 - TABLE 1 STABILITY COMPARISON OF STATE REPRESENTATIONS)
mean_q_value = np.mean(q_values) # Mean Q-value
std_dev_q = np.std(q_values) # Standard Deviation
variance_q = np.var(q_values) # Variance
range_q = np.max(q_values) - np.min(q_values) # Range (Max - Min Q-Value)
# Print results for each metric
print(f"Mean Q-value: {mean_q_value}")
print(f"Standard Deviation: {std_dev_q}")
print(f"Variance: {variance_q}")
print(f"Range (Max - Min Q-Value): {range_q}")



# Assuming 'rewards' contains per-episode rewards, and 'q_values' stores Q-values (SOP 2)
num_bins = 10  # Define bins for heatmap visualization

def plot_reward_distribution(rewards):
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, bins=num_bins, kde=True, color='purple')
    plt.xlabel('Reward Values')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution: Impact of Binary Satisfaction')
    plt.savefig('binary_reward_satisfaction.png')  # Save plot as PNG
    plt.show()




# Assuming q_values is a NumPy array storing sampled Q-values across states
plot_reward_distribution(rewards)
