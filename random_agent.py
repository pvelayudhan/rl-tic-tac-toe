from tictactoe import Game
import numpy as np
import random

class RandomAgent:
    def play(self, game):
        if game.done is True:
            return None
        legal_moves = np.where(np.reshape(game.squares, [9]) == 0)[0]
        legal_move_indices = np.arange(legal_moves.size)
        chosen_move = legal_moves[np.random.choice(legal_move_indices)]
        formatted_move = np.zeros(9, dtype=float)
        formatted_move[chosen_move] = 1
        formatted_move = np.reshape(formatted_move, [3, 3])
        rank, file = np.where(formatted_move == 1)
        game.play(rank[0], file[0])
        return None
