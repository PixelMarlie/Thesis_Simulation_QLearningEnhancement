from QLearningSimEnhanceENV import SocialLearningEnv, QLearningAgent
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

# Training setup
num_actions = 5  # The agent recommends up to 5 topics
env = SocialLearningEnv(num_actions=num_actions)
agent = QLearningAgent(input_dim=env.num_features, num_actions=num_actions)

# Training parameters
num_episodes = 100
rewards = []
dynamic_rewards = []  # Store dynamic reward satisfaction values
start_time = time.time()

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        step_count += 1
        dynamic_rewards.append(reward)  # Store dynamic reward values

        if step_count >= 1000:  # Prevent infinite loops
            done = True

    rewards.append(total_reward)
    agent.decay_epsilon()
    print(f"Episode {episode}: Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# Training complete
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Reward trend visualization
plt.plot(rewards, label='Unenhanced Q-Learning Rewards', color='green')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Reward Trend')
plt.savefig('reward_trend.png')

# Visualize Q-values for sampled states
sampled_states = [env.reset() for _ in range(100)]
q_values = []

for state in sampled_states:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values.append(agent.q_network(state_tensor).numpy())

q_values = np.array(q_values)  # Converts the list to a NumPy array
q_values = q_values.squeeze()  # New shape: (num_sampled_states, num_actions)

# Plot Q-values
plt.figure(figsize=(10, 6))
for action in range(num_actions):
    plt.plot(q_values[:, action], label=f'Action {action}')
plt.xlabel('Sampled States')
plt.ylabel('Q-values')
plt.title('Learned Q-Values for Sampled States')
plt.legend()
plt.savefig('q_values_plot.png')

# Histogram of Dynamic Reward Satisfaction
plt.figure(figsize=(10, 6))
sns.histplot(dynamic_rewards, color="blue", label="Dynamic Reward Satisfaction", kde=True, bins=20, alpha=0.6)
plt.xlabel("Reward Values")
plt.ylabel("Frequency")
plt.title("Histogram of Dynamic Reward Satisfaction in Enhanced Q-Learning")
plt.legend()
plt.savefig('dynamic_reward_satisfaction.png')
plt.show()
