from QLearningSimEnhanceENV import SocialLearningEnv, QLearningAgent
import time
import matplotlib.pyplot as plt
import torch
import psutil
import numpy as np
import seaborn as sns
from scipy.stats import kurtosis

# Training setup
num_actions = 5  # The agent recommends up to 5 topics
env = SocialLearningEnv(num_actions=num_actions)
agent = QLearningAgent(input_dim=env.num_features, num_actions=num_actions)

state_tensor = torch.randn(1, env.num_features)  # Random input

start_time = time.time()
_ = agent.q_network(state_tensor)

# Training parameters
num_episodes = 10
rewards = []
dynamic_rewards = []  # Store dynamic reward satisfaction values

# SOL 3 MEMORY EFFICIENCY OF SIMPLIFIED NN FOR Q-LEARNING
memory_usage_nn = []  # Store memory usage for NN
episodes = list(range(0, num_episodes+1, 5))  # Matches memory logging frequency (5 episodes)
cpu_usage = []
gpu_usage = []

# Define state-action tracking (sparsity heatmap)
num_states = env.num_features  # Or the state space size if discrete
num_actions = agent.num_actions
state_action_counts = np.zeros((num_states, num_actions))  # Initialize tracking

# Track peak memory usage (SOL 3)
peak_memory_nn = 0

# Training Loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    start_cpu = psutil.cpu_percent()
    start_gpu = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB

    while not done:
        action = agent.choose_action(state)

        # Track action selection frequency
        state_index = np.argmax(state)  # Use an appropriate method for indexing
        state_action_counts[state_index, action] += 1  # Log action choice

        next_state, reward, done, _ = env.step(action)

        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        step_count += 1
        dynamic_rewards.append(reward)  # Store dynamic reward values

        if step_count >= 1000:  # Prevent infinite loops
            done = True

    cpu_usage.append(psutil.cpu_percent() - start_cpu)
    gpu_usage.append((torch.cuda.memory_allocated() / (1024 * 1024)) - start_gpu)

    # Log memory consumption (SOL 3)
    if episode % 5 == 0 or episode == num_episodes - 1:  # Ensure last episode is logged
        # Get memory usage (for both approaches)
        process = psutil.Process()
        memory_used = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        memory_usage_nn.append(memory_used)

        # Update peak memory usage
        peak_memory_nn = max(peak_memory_nn, memory_used)

    rewards.append(total_reward)
    agent.decay_epsilon()

    print(f"Episode {episode}: Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# Normalize the frequency counts
state_action_counts_normalized = state_action_counts / np.max(state_action_counts)

# Training complete
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

forward_pass_time = (end_time - start_time) * 1000  # Convert to ms
print(f"Forward Pass Time: {forward_pass_time:.4f} ms")

# Plotting CPU & GPU Usage (SOL 3)
plt.plot(cpu_usage, label="CPU Usage (%)", color='red')
plt.plot(gpu_usage, label="GPU Memory (MB)", color='blue')
plt.xlabel("Episodes")
plt.ylabel("Resource Usage")
plt.title("CPU & GPU Memory Usage Over Training")
plt.legend()
plt.show()

# Reward trend visualization
plt.plot(rewards, label='Unenhanced Q-Learning Rewards', color='green')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Reward Trend')
plt.savefig('reward_trend.png')

# Visualize Q-values for sampled states (SOL 1)
sampled_states = [env.reset() for _ in range(100)]
q_values = []

for state in sampled_states:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values.append(agent.q_network(state_tensor).numpy())

q_values = np.array(q_values)  # Converts the list to a NumPy array
q_values = q_values.squeeze()  # New shape: (num_sampled_states, num_actions)

# Plot Q-values (SOL 1)
plt.figure(figsize=(10, 6))
for action in range(num_actions):
    plt.plot(q_values[:, action], label=f'Action {action}')
plt.xlabel('Sampled States')
plt.ylabel('Q-values')
plt.title('Learned Q-Values for Sampled States')
plt.legend()
plt.savefig('q_values_plot.png')

# (SOP1 - TABLE 1 STABILITY COMPARISON OF STATE REPRESENTATIONS)
mean_q_value = np.mean(q_values) # Mean Q-value
std_dev_q = np.std(q_values) # Standard Deviation
variance_q = np.var(q_values) # Variance
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

# SOP 3 - TABLE 3 MEMORY EFFICIENCY & SPARSITY COMPARISON
# Final memory usage after training
final_memory_nn = memory_usage_nn[-1]
# Calculate sparsity score: Ratio of neurons never activated (approximated by action selection)
total_actions = np.prod(state_action_counts.shape)
zero_actions = total_actions - np.count_nonzero(state_action_counts)
sparsity_score_nn = (zero_actions / total_actions) * 100
# Efficiency Ratio (final reward per MB of memory used)
efficiency_ratio_nn = rewards[-1] / final_memory_nn if final_memory_nn > 0 else 0
# Print Table 3 results
print("\nTABLE 3:")
print(f"Peak Memory Usage: {peak_memory_nn:.2f} MB")
print(f"Final Memory Usage: {final_memory_nn:.2f} MB")
print(f"Sparsity Score: {sparsity_score_nn:.2f}%")
print(f"Efficiency Ratio (Final Reward / Memory): {efficiency_ratio_nn:.4f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(agent.q_network)
print(f"Total Trainable Parameters: {num_params}")

def layer_memory_usage(model):
    for name, param in model.named_parameters():
        mem = param.numel() * param.element_size() / (1024 * 1024)  # Convert to MB
        print(f"Layer: {name} | Memory Usage: {mem:.4f} MB")


# Histogram of Dynamic Reward Satisfaction (SOL 2)
plt.figure(figsize=(10, 6))
sns.histplot(dynamic_rewards, color="blue", label="Dynamic Reward Satisfaction", kde=True, bins=20, alpha=0.6)
plt.xlabel("Reward Values")
plt.ylabel("Frequency")
plt.title("Histogram of Dynamic Reward Satisfaction in Enhanced Q-Learning")
plt.legend()
plt.savefig('dynamic_reward_satisfaction.png')
plt.show()

# Simplified NN Memory Usage (SOL 3)
print(len(episodes), len(memory_usage_nn))  # Both should be equal
plt.plot(episodes, memory_usage_nn, 'b-s', label="Neural Network Q-learning")
plt.xlabel("Episodes")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Growth: Simplified Neural Network in Social Learning")
plt.legend()
plt.savefig('simplified_NN_growth.png')
plt.show()

# Generate Sparsity Heatmap (SOL 3)
plt.figure(figsize=(10, 6))
sns.heatmap(state_action_counts_normalized, cmap="Blues", cbar=True)
plt.xlabel("Actions")
plt.ylabel("States")
plt.title("Sparsity Heatmap: NN-based Q-learning Action Selection")
plt.savefig("nn_sparsity_heatmap.png")
plt.show()