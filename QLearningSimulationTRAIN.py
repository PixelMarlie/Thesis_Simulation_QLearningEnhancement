from QLearningSimulationENV import SocialLearningEnv, QLearningAgent
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
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

# Track peak memory usage (SOP 3)
peak_memory_q_table = 0

env = SocialLearningEnv(num_actions, num_preferences)
agent = QLearningAgent(num_preferences=num_preferences, action_space=num_actions)

# Training parameters
num_episodes = 1000  # Reduce for quicker comparison
rewards = []
start_time = time.time()

# SOP 3 MEMORY EFFICIENCY OF Q-TABLE FOR Q-LEARNING
memory_usage_q_table = []  # Store memory usage over episodes for Q-table
episodes = list(range(0, num_episodes+1, 5))  # Matches memory logging frequency (5 episodes)

# For visualizing table sparsity (SOP 3)
q_table = agent.get_q_table()

# For visualizing growth of q-table over time (SOP 3)
nonzero_q_values_history = []

# Traininig Loop
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

    # Log memory consumption (SOP 3)
    if episode % 5 == 0 or episode == num_episodes - 1:  # Ensure last episode is logged
        # Get memory usage (for both approaches)
        process = psutil.Process()
        memory_used = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        memory_usage_q_table.append(memory_used)

        # Update peak memory usage
        peak_memory_q_table = max(peak_memory_q_table, memory_used)

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
std_dev_q = np.std(q_values) # Standard Deviation of Q-values
variance_q = np.var(q_values) # Variance of Q-values
range_q = np.max(q_values) - np.min(q_values) # Range (Max - Min Q-Value)
# Print results for each metric
print("\nTABLE 1: ")
print(f"Mean Q-value: {mean_q_value}")
print(f"Standard Deviation: {std_dev_q}")
print(f"Variance: {variance_q}")
print(f"Range (Max - Min Q-Value): {range_q}")

# (SOP 2 - TABLE 2 REWARD TREND METRICS COMPARISON: TRADITIONAL VS ENHANCED Q-LEARNING)
mean_reward = np.mean(rewards) # Mean Reward
median_reward = np.median(rewards) # Median Reward
variance_reward = np.var(rewards) # Variance of Rewards
std_dev_reward = np.std(rewards) # Standard Deviation of Rewards
max_reward = np.max(rewards) # Max of Rewards
cumulative_rewards = np.sum(rewards) # Cumulative Rewards (Total)
# Reward Growth Rate
trend_slope = np.polyfit(range(len(rewards)), rewards, 1)[0]
# Kurtosis via Scipy library
kurt = kurtosis(rewards)
# Print results for each metric
print("\nTABLE 2: ")
print(f"Mean Reward: {mean_reward}")
print(f"Median Reward: {median_reward}")
print(f"Variance Reward: {variance_reward}")
print(f"Standard Deviation of Rewards: {std_dev_reward}")
print(f"Max of Rewards: {max_reward}")
print(f"Cumulative Rewards: {cumulative_rewards}")
print(f"Reward Growth Rate: {trend_slope}")
print(f"Kurtosis of Rewards: {kurt}")

# (SOP 3 - TABLE 3 MEMORY EFFICIENCY AND SPARSITY COMPARISON)
# Final memory usage after training
final_memory_q_table = memory_usage_q_table[-1]
# TABLE 3 Calculate sparsity score (percentage of zero entries in Q-table)
q_table_size = np.prod(agent.get_q_table().shape)
zero_entries = q_table_size - np.count_nonzero(agent.get_q_table())
sparsity_score_q_table = (zero_entries / q_table_size) * 100
# TABLE 3 Efficiency Ratio (final reward per MB of memory used)
efficiency_ratio_q_table = rewards[-1] / final_memory_q_table if final_memory_q_table > 0 else 0
# Print Table 3 results
print("\nTABLE 3:")
print(f"Peak Memory Usage: {peak_memory_q_table:.2f} MB")
print(f"Final Memory Usage: {final_memory_q_table:.2f} MB")
print(f"Sparsity Score: {sparsity_score_q_table:.2f}%")
print(f"Efficiency Ratio (Final Reward / Memory): {efficiency_ratio_q_table:.4f}")


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


# Q-TABLE Memory Usage (SOP 3)
plt.plot(episodes, memory_usage_q_table, 'b-s', label="Q-table Q-learning")
plt.xlabel("Episodes")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Growth: Q-table in Social Learning")
plt.legend()
plt.savefig('Qtable_growth.png')
plt.show()

# Assuming q_values is a NumPy array storing sampled Q-values across states
plot_reward_distribution(rewards)
