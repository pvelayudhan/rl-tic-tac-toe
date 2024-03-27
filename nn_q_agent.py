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
S = {}

# Function to choose action based on epsilon-greedy policy
def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # Random action
    else:
        return np.argmax(q_values)  # Greedy action

def q_learning_play(game, qtable, states_table):
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
    states_table[game.board.calculate_hash()] = possible_states
    x, y, _ = possible_states[picked_value]
    return(x, y, qtable, states_table)

def undo_hash(game_hash):
    # Convert the integer hash to a string
    game_hash_str = str(game_hash)
    # Insert spaces to separate the digits
    game_hash_str_spaced = ' '.join(game_hash_str)
    # Convert the string back to a numpy array
    game_hash_array = np.fromstring(game_hash_str_spaced, dtype=int, sep=' ')
    while len(game_hash_array) < 9:
        game_hash_array = np.concatenate(([0], game_hash_array))
    # Reshape the numpy array to its original shape (3x3)
    original_shape = (3, 3)
    original_array = np.reshape(game_hash_array, original_shape)
    return original_array

def extract_max_possible_q(state, Q):
    squares = undo_hash(state)
    # Setting up the game from the given state
    game = Game()
    game.board.squares = squares
    game.turn = np.sum(game.board.squares != 0)
    # Explore possible next states
    max_Q = 0
    for x in range(3):
        for y in range(3):
            if game.board.squares[x][y] == 0:
                game_copy = deepcopy(game)
                game_copy.play(x, y)
                possible_state = game_copy.board.calculate_hash()
                if possible_state in Q:
                    if Q[possible_state] > max_Q:
                        max_Q = Q[possible_state]
                else:
                    Q[possible_state] = 0.5
                    if Q[possible_state] > max_Q:
                        max_Q = Q[possible_state]
    return max_Q

# 1. Keep track of all the selected states
# 2. The very last state gets the player reward from the game as its value
# 3. Step backwards to the 2nd last state. Now identify, from here, of all the next possible states, the
#    highest Q-value (make fn for this). This is max_a Q(S', a).
# 4. Plug that value into the Q update formula:
#    - (1 - learning rate) * the old Q + learning rate * disc fac * max_a Q(S', a)
# 5. Repeat backwards through all the states that were selected in the game

# put all this inside 100 games e.g.
###############################################################################

Q = {}
rewards = list()
gamma = 0.95
alpha = 0.95

for i in range(10000):
    print(f"Game {i}")
    random_agent = RandomAgent()
    visited_states = list()
    game = Game()
    while not game.done:
        # Q-agent is P1
        x, y, Q, S = q_learning_play(game, Q, S)
        game.play(x, y)
        # game.board.show()
        visited_state = game.board.calculate_hash()
        visited_states.append(visited_state)
        if game.done:
            break
        # Random Agent is P2
        random_agent.play(game)
        #print("Random agent turn:")
        #game.board.show()
    # Keep track of q-agent's rewards for our own monitoring
    rewards.append(game.p1_reward)
    # Update Q table by working backwards through visited states
    rev_states = visited_states[::-1]
    for i in range(len(rev_states)):
        state = rev_states[i]
        if i == 0:
            Q[state] = game.p1_reward
        else:
            max_next_q = extract_max_possible_q(state, Q)
            Q[state] = ((1 - gamma) * Q[state]) + gamma * alpha * max_next_q

Q2 = {}
# Check Q-agent performance vs. random agent after 100 games
evaluation_rewards = list()
for i in range(100):
    random_agent = RandomAgent()
    game = Game()
    while not game.done:
        # Q-agent is P1
        x, y, Q2, S = q_learning_play(game, Q2, S)
        game.play(x, y)
        # game.board.show()
        if game.done:
            break
        # Random Agent is P2
        random_agent.play(game)
    # Keep track of q-agent's rewards for our own monitoring
    evaluation_rewards.append(game.p1_reward)
np.mean(evaluation_rewards)
