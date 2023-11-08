from tictactoe import Game
import numpy as np
from copy import deepcopy

game = Game()

game.squares

game.board.calculate_hash()

# Q value is equal to

q_table = {}

def play(game):
    # If the game is finished, do nothing
    if game.done:
        return None
    # If the game is not finished, find the winner value of all the possible
    #  versions of the game following one more turn
    else:
        possible_moves = np.where(game.board.squares == 0)
        possible_moves = np.transpose(possible_moves)
        possible_values = list()
        for possible_move in possible_moves:
            possible_game = deepcopy(game)
            possible_game.play(possible_move[0], possible_move[1])
            possible_values.append(self.play(possible_game, main_call=False))
        if not main_call:
            if game.turn % 2 == 0:
                return max(possible_values)
            else:
                return min(possible_values)
        else:
            best_value_pos = possible_values.index(max(possible_values))
            best_move = possible_moves[best_value_pos]
            game.play(best_move[0], best_move[1])
