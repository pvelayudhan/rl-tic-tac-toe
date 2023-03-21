from tictactoe import Game
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(9, activation="sigmoid")
])


#|%%--%%| <0uaRz4M2ub|nBwOuE3A7p>


game = Game()

game.play(2, 0)

game.board.show()


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
        square_probs = model(formatted_board)
        # have a function here to remove illegal moves
        # have a function here to probabilistically choose an action
        loss = tf.reduce_mean(loss_fn(target_square, square_probs))
    grads = tape.gradient(loss, model.trainable_variables)
    # function here to convert action into rank and file
    rank, file = square_probs_to_action(square_probs)
    state, reward, done = game.play(rank, file)
    return state, reward, done, grads
