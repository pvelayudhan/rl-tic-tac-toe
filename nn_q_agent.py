from tictactoe import Game
import numpy as np
#import tensorflow as tf

# The Q-table
Q = {}

# Q(S, A) = gamma * max_a Q(S', a)

game = Game()

game.board.show()

game.play(1, 1)

game.winner

"""
1. Look up the board state in the Q-table:
    - If the state exists -> ?pick the highest Q-value state OR a random state (based on epsilon)
        - If there are ties, pick randomly amongst the ties
    - If the state doesn't exist -> pick any action randomly
2. Keep track of all the state-action pairs that were chosen based on above
3. Update all those state-action pairs based on the fat Q formula
"""

# Function to choose action based on epsilon-greedy policy
def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # Random action
    else:
        return np.argmax(q_values)  # Greedy action
























"""
# Define a Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(9,)),  # Input layer with ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with ReLU activation
    tf.keras.layers.Dense(1)  # Output layer (no activation for regression tasks)
])

model.predict(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(1, 9))

model.predict(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))


game


state = game.board.calculate_hash()

print(state)

Q[(state, state)] = 3
"""
