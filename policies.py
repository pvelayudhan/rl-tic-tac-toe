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
game.play(1, 2)
game.play(0, 1)
game.play(2, 2)
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
    return formatted_move


def play_one_move(game, model, loss_fn):
    with tf.GradientTape() as tape:
        formatted_board = game.board.squares.reshape(1, 9)
        # 1. Get a prediction from model()
        square_probs = model(formatted_board)
        # 2. Pick an action that can be fed into environment
        target_square = choose_move(game, square_probs)
        # Switch the above^ to pull out formatted move as a square, add
        #  a new function that makes square into rank and file
        # Next things to do
        loss = tf.reduce_mean(loss_fn(target_square, square_probs))
    grads = tape.gradient(loss, model.trainable_variables)
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

#|%%--%%| <6rPK8eNKTp|wrtgaKWxPI>


loss_fn = tf.keras.losses.categorical_crossentropy
loss_fn = tf.keras.losses.binary_crossentropy
