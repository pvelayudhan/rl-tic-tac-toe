from tictactoe import Game
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(9, activation="sigmoid")
])


#|%%--%%| <0uaRz4M2ub|nBwOuE3A7p>


game = Game()

# Win sequence
game.play(0, 2)
game.play(1, 2)  # bot
game.play(0, 1)
game.play(2, 2)  # bot
game.play(0, 0)

# Draw sequence
game.play(0, 2)
game.play(1, 1)
game.play(0, 1)
game.play(0, 0)
game.play(2, 2)
game.play(1, 2)
game.play(1, 0)
game.play(2, 1)
game.play(2, 0)


#|%%--%%| <nBwOuE3A7p|6rPK8eNKTp>

# https://stackoverflow.com/a/38250088/16626788
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# This returns a legal, soft-max'd move
def choose_move(game, square_probs):
    legal_moves = np.where(np.reshape(game.squares, [9]) == 0)[0]
    legal_move_chosen = False
    soft_probs = softmax(tf.reshape(square_probs, [9]))
    while legal_move_chosen is False:
        chosen_move = np.random.choice(np.arange(9), 1, p=soft_probs)[0]
        if chosen_move in legal_moves:
            legal_move_chosen = True
    formatted_move = np.zeros(9, dtype=float)
    formatted_move[chosen_move] = 1
    formatted_move = tf.convert_to_tensor(formatted_move[np.newaxis])
    return formatted_move


# Randomly select a legal move
def random_move(game):
    legal_moves = np.where(np.reshape(game.squares, [9]) == 0)[0]
    legal_move_chosen = False
    while legal_move_chosen is False:
        chosen_move = np.random.choice(np.arange(9), 1)[0]
        if chosen_move in legal_moves:
            legal_move_chosen = True
    formatted_move = np.zeros(9, dtype=float)
    formatted_move[chosen_move] = 1
    return formatted_move



def play_one_move(game, model, loss_fn):
    with tf.GradientTape() as tape:
        # 1. extract a state that can be fed into model()
        formatted_board = game.board.squares.reshape(1, 9)
        # 2. Get a prediction from model()
        square_probs = model(formatted_board)
        # 3. Pick an action that can be fed into environment
        target_square = choose_move(game, square_probs)
        #print(square_probs)
        # 4. Define the loss(?)
        loss = tf.reduce_mean(loss_fn(target_square, square_probs))
    # Calculate the gradients between predicted and actual(?)
    grads = tape.gradient(loss, model.trainable_variables)
    # Actually play the selected move, collecting the reward and done state
    target_square = np.reshape(target_square, [3, 3])
    rank, file = np.where(target_square == 1)
    reward, done = game.play(rank[0], file[0])
    return reward, done, grads


# The HOML3 function
def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        # 1. (DONE) Get a prediction from model()
        left_proba = model(obs[np.newaxis])
        # Generate one one-sized array of uniform numbers and compare with left
        #  probability. The action is false if action is false
        # 2. (DONE) Pick an action that can be fed into environment
        action = (tf.random.uniform([1, 1]) > left_proba)
        # If the action is false (0), the target probability of going left is 1
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, truncated, _ = env.step(int(action))
    return obs, reward, done, truncated, grads

#|%%--%%| <6rPK8eNKTp|SzXI15gBzx>


loss_fn = tf.keras.losses.categorical_crossentropy

# loss_fn = tf.keras.losses.binary_crossentropy


# y = mx0 + b -> y' = m = -3
# (how bad is the neural net doing) = m(weight of neural net node) + b

"""
y = mx + b

1. make a list of probabilities of moves using neural net
2. using the probabilities, we pick an action
3. in gradient tape, we keep a track of the gradients needed to make the neural net more or less likely to take our picked action
4. we play some games
5. in the games we won, we apply the gradients that make the moves we picked more likely
6. in the games we lost, we apply the gradients that make the moves we picked less likely

# EXAMPLE:
1. neural net: y = 3x (+ a die roll)
2. probabilities given by NN for the state (x = 2): y = 6 (+ die roll) = anything from 3 to 9
3. we picked 3
4. INSIDE THE GRADIENT TAPE:
    y = 3x + b -> y' = 3 -> raising m will raise y, lowering m will lower y
    (y is the predicted probability)
5. we play the game
6. Case 1: we win the game!
    6a: look back at the "loss" between our output probabilities and our action:
        we predicted ~6, we picked 3, we won.
        we want our predictions to have been EVEN LOWER so that our chance of picking 3 will go up next time.
        to make our prediction (6) more like our winning answer (3), we change the NN weight based on our derivative: based on gradient tape (our memory), we should LOWER m so that y also becomes closer to 3
    6b: new neural net is y = 2.5x + b

Repeat!

"""
#|%%--%%| <SzXI15gBzx|1sGtfgIBy2>

# Each step = one turn of tic-tac-toe
# Each "episode" = one whole game of tic-tac-toe


game = Game()

def play_multiple_episodes(game, n_episodes, model, loss_fn, n_max_steps=9):
    all_rewards = []
    all_grads = []
    #1. Loop through many games of ttt
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        #2. Initialize a new game
        #obs, _ = env.reset()
        #print(f"Playing game #{episode}")
        game = Game()
        for _ in range(n_max_steps):
            #obs, reward, done, truncated, grads = play_one_step(env, obs, model, loss_fn)
            reward, done, grads = play_one_move(game, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    # all_rewards is a list containing a list of rewards for each episode
    # all_grads is a list containing a list of gradients for each episode
    return all_rewards, all_grads


# A discounted reward is the reward for the current step plus the reward for
#  all future steps multiplied by the disocunt factor. For example, if our
#  rewards were [2, 4, 8] and we had a discount factor of 0.5, our discounted
#  rewards would be [(2 + 8*0.5 =) 6, (4 + 8*0.5 =) 8, 8]. Note that this
#  calculation is done starting at the last reward and working backwards.
def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    # Range params are start index, end index, and step size
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discount_factor * discounted[step + 1]
    return discounted


# This function performs standard normalization across ALL rewards
def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]



n_iterations = 10
n_episodes_per_update = 2
discount_factor = 0.95


#|%%--%%| <1sGtfgIBy2|kbUk7tZ9l2>


model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(9, activation="sigmoid")
])

optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)

#|%%--%%| <kbUk7tZ9l2|lCUWqS6p3n>



for iteration in range(n_iterations):
    print(f"{iteration + 1} / {n_iterations}")
    all_rewards, all_grads = play_multiple_episodes(game, n_episodes_per_update, model, loss_fn)
    # extra code – displays some debug info during training
    total_rewards = sum(map(sum, all_rewards))
    print(f"\rIteration: {iteration + 1}/{n_iterations},"
          f" mean rewards: {total_rewards / n_episodes_per_update:.1f}", end="")

    all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                       discount_factor)
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

#|%%--%%| <lCUWqS6p3n|dFrmdtjX0e>



game = Game()

game.board.show()

play_one_move(game, model, loss_fn)

game.board.show()

game.play(1, 1)

game.board.show()

play_one_move(game, model, loss_fn)

game.board.show()

game.play(2, 2)

play_one_move(game, model, loss_fn)

game.board.show()

game.play(2, 1)

play_one_move(game, model, loss_fn)

game.board.show()


game.play(0, 2)


play_one_move(game, model, loss_fn)


