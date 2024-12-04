from flask import Flask, render_template, redirect, url_for
import random

app = Flask(__name__)

# Initialize the Q-table and user preferences (simple example)
q_table = {}
user_preference = 2  # Simulated user preference (can be any value between 1 and 5)

# Example: Define your actions (recommendations)
actions = [1, 2, 3, 4, 5]  # Possible recommendation IDs


def initialize_q_table():
    global q_table
    # Initialize Q-table: states x actions
    for state in range(1, 6):  # States are user preferences (1 to 5)
        q_table[state] = {action: 0 for action in actions}


def get_reward(user_pref, recommended):
    # Reward is higher if recommendation matches user preference
    if user_pref == recommended:
        return 1  # Positive reward for a match
    else:
        return -2  # Negative reward for a mismatch


def choose_action(state):
    # Choose action (recommendation) based on the current Q-table for that state
    return max(q_table[state], key=q_table[state].get)


def update_q_table(state, action, reward, next_state):
    # Q-learning update rule
    learning_rate = 0.1
    discount_factor = 0.9
    max_next_q_value = max(q_table[next_state].values())
    q_table[state][action] += learning_rate * (reward + discount_factor * max_next_q_value - q_table[state][action])


@app.route('/')
def home():
    global user_preference
    return render_template('home.html', q_table=q_table, user_preference=user_preference)


@app.route('/automate')
def automate():
    total_rewards = 0
    num_episodes = 1000  # Number of episodes to simulate

    # Simulate the episodes
    for episode in range(num_episodes):
        state = random.choice(range(1, 6))  # Randomly pick a user preference state (1 to 5)
        action = choose_action(state)  # Choose a recommendation based on Q-table
        reward = get_reward(state, action)  # Get reward based on matching or not
        next_state = random.choice(range(1, 6))  # Randomly pick the next state (next user preference)

        # Update the Q-table based on the reward
        update_q_table(state, action, reward, next_state)

        total_rewards += reward  # Accumulate total reward

    return render_template('automate_results.html', total_rewards=total_rewards)


if __name__ == '__main__':
    initialize_q_table()  # Initialize Q-table when the app starts
    app.run(debug=True)
