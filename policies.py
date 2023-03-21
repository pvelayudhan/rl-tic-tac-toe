import tensorflow as tf

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
