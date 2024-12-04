from QLearningSimulationENV import SocialLearningEnv, QLearningAgent
import time
import numpy as np
import matplotlib.pyplot as plt

# Training setup
num_actions = 5  # Application will recommend up to 5 topics
num_preferences = 5  # User sets social learning preferences
env = SocialLearningEnv(num_actions, num_preferences)
agent = QLearningAgent(num_preferences=num_preferences, action_space=num_actions)

# Training parameters
num_episodes = 10  # Reduce for quicker comparison
rewards = []
start_time = time.time()

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

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

    rewards.append(total_reward)  # Track total reward for this episode
    agent.decay_epsilon()
    print(f"Episode {episode}: Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# Training complete
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Reward trend visualization
plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Unenhanced Q-Learning Rewards', color='blue')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Unenhanced Q-Learning: Training Reward Trend')
plt.legend()
plt.savefig('unenhanced_reward_trend.png')  # Save plot as PNG
plt.show()

# Visualize Q-values for sampled states
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
