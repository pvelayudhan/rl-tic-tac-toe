from tictactoe import Game
import numpy as np
import random
import tensorflow as tf

class RandomAgent:
    def play(self, game):
        if game.done is True:
            return None
        legal_moves = np.where(np.reshape(game.squares, [9]) == 0)[0]
        legal_move_chosen = False
        while legal_move_chosen is False:
            chosen_move = np.random.choice(np.arange(9), 1)[0]
            if chosen_move in legal_moves:
                legal_move_chosen = True
        formatted_move = np.zeros(9, dtype=float)
        formatted_move[chosen_move] = 1
        formatted_move = tf.convert_to_tensor(formatted_move[np.newaxis])
        target_square = np.reshape(formatted_move, [3, 3])
        rank, file = np.where(target_square == 1)
        game.play(rank[0], file[0])
        return None
