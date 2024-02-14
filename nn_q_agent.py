from tictactoe import Game
from random_agent import RandomAgent
import numpy as np
#import tensorflow as tf
from copy import deepcopy

"""
1. (done) Convert the given board into board state
2. (done) Look up the board state in the Q-table:
    - (done) If the state exists -> ?pick the highest Q-value state OR a random state (based on epsilon)
        - (done) If there are ties, pick randomly amongst the ties
    - (done) If the state doesn't exist -> initializing with 0.5
3. Play the move

...?

3. Keep track of all the state-action pairs that were chosen based on above
4. Update all those state-action pairs based on the fat Q formula

Moves that were never played: we choose initial value
Moves that are played: keep track of them, update with Q alg at end of each game
"""

# The Q-table
Q = {}


game = Game()

game.board.show()

game.play(1, 1)

game.play(1, 2)

game.play(0, 2)

game.board.show()

game.board.calculate_hash()

# Function to choose action based on epsilon-greedy policy
def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # Random action
    else:
        return np.argmax(q_values)  # Greedy action

def q_learning_play(game, qtable):
    game.board.show()
    possible_states = list()
    # Make hashes of each possible next game state
    for x in range(3):
        for y in range(3):
            #print(f"x: {x} | y: {y} | value: {game.board.squares[x][y]}")
            if game.board.squares[x][y] == 0:
                game_copy = deepcopy(game)
                game_copy.play(x, y)
                possible_states.append((x, y, game_copy.board.calculate_hash()))
    # Get q_values of all the possible next game states from Q table
    # or initialized with 0.5 if the state was never seen before
    q_values = list()
    for state in possible_states:
        state_hash = state[2]
        if state_hash in qtable:
            q_values.append(qtable[state_hash])
        else:
            q_values.append(0.5) # our initial Q values
            qtable[state_hash] = 0.5
    # Use epsilon greedy to make a move
    picked_value = epsilon_greedy_policy(q_values, 0.1)
    x, y, _ = possible_states[picked_value]
    return(x, y, qtable)


visited_states = list()
rewards = list()

# Play through a whole game

game = Game()
game.board.show()

random_agent = RandomAgent()


game.board.show()

while not game.done:
    # Q-agent is P1
    x, y, Q = q_learning_play(game, Q)
    game.play(x, y)
    # Random Agent is P2
    random_agent.play(game)
# Store visited states of that game and associated rewards

game.board.show()


"""
At the end of each game update the Q value of all moves in the game according to the game result.
For a win we will award a reward of 1 to the last move, for a loss a reward of 0 and for a draw we will give a reward of 0.5.

The final move will get the reward as its new Q value. For all the other moves in that game we will use the following formula 

Q(S, A) = gamma * max_a Q(S', a)

Q(S, A) = (1 - alpha) * Q(S, A) + alpha*gamma*(Q value of the best state following S)

S can lead to:
    Sa OR
    Sb OR
    Sc
And Sb is the highest value state out of Sa, Sb, Sc, then

""" 




game.play(1, 1)

game.winner



## Function to create the Q-network
#def create_dueling_q_network(input_shape, output_size):
#    # Define the input layer
#    input_layer = tf.keras.layers.Input(shape=input_shape)
#    # Shared dense layers for the value and advantage streams
#    shared_dense = tf.keras.layers.Dense(32, activation='relu')(input_layer)
#    # Value stream
#    value_stream = tf.keras.layers.Dense(1)(shared_dense)
#    # Advantage stream
#    advantage_stream = tf.keras.layers.Dense(output_size)(shared_dense)
#    # Combine value and advantage streams to get Q-value
#    q_values = value_stream + (advantage_stream - tf.reduce_mean(advantage_stream, axis=1, keepdims=True))
#    model = tf.keras.models.Model(inputs=input_layer, outputs=q_values)
#    return model























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
