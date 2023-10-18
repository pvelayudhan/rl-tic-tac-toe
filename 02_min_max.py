from tictactoe import Game
import random
import numpy as np
from copy import deepcopy


def min_max(game, main_call=True):
    # If the game is finished, return the winner value
    if game.done:
        if game.winner == None: # game was a draw
            return 0
        elif game.winner == 1: # player 1 wins
            return 1
        elif game.winner == 2: # player 2 wins
            return -1
    # If the game is not finished, find the winner value of all the possible
    #  versions of the game following one more turn
    else:
        possible_moves = np.where(game.board.squares == 0)
        possible_moves = np.transpose(possible_moves)
        # possible_values will store the winner value resulting from playing each
        #  possible move
        possible_values = list()
        # it's player 1 (X)'s turn: find move returning max game value
        if game.turn % 2 == 0:
            for possible_move in possible_moves:
                possible_game = deepcopy(game)
                possible_game.play(possible_move[0], possible_move[1])
                possible_values.append(min_max(possible_game, main_call=False))
            # if this isn't the main (first) call of the function, we are just
            #  trying to tell our parent call what the winner value is of the
            #  possible game they provided us
            if not main_call:
                return max(possible_values)
            # if this is the main (first) call of the function, actually play
            #  a move yielding optimal value out of all the possible next moves
            else:
                best_value_pos = possible_values.index(max(possible_values))
                best_move = possible_moves[best_value_pos]
                game.play(best_move[0], best_move[1])
        # it's player 2 (O)'s turn: find move returning min game value
        else:
            for possible_move in possible_moves:
                possible_game = deepcopy(game)
                possible_game.play(possible_move[0], possible_move[1])
                possible_values.append(min_max(possible_game, main_call=False))
            # if this isn't the main (first) call of the function, we are just
            #  trying to tell our parent call what the winner value is of the
            #  possible game they provided us
            if not main_call:
                return min(possible_values)
            # if this is the main (first) call of the function, actually play
            #  a move yielding optimal value out of all the possible next moves
            else:
                best_value_pos = possible_values.index(min(possible_values))
                best_move = possible_moves[best_value_pos]
                game.play(best_move[0], best_move[1])


# A drawn game
game = Game()

game.play(0, 0)
game.play(0, 1)
game.play(1, 2)
game.play(1, 0)
game.play(1, 1)
game.play(0, 2)
game.play(2, 0)
game.play(2, 2)
game.play(2, 1)

game.board.show()

min_max(game)


# A drawn game
game = Game()

game.play(0, 0)
game.play(0, 1)
game.play(1, 2)
game.play(1, 0)
game.play(1, 1)
game.play(0, 2)
game.play(2, 0)
#game.play(2, 2)
# game.play(2, 1)

game.board.show()

min_max(game)

game.board.show()

game.turn % 2 == 0

game.board.squares


possible_moves = np.where(game.board.squares == 0)
possible_moves = np.transpose(possible_moves)

possible_moves[0]


print(move_values)

move_values = list()
move_values.append(5)
move_values.append(5)
move_values.append(5)
move_values.append(3)

move_values

move_values.index(min(move_values))

move_values.index(max(move_values))

for possible_move in possible_moves:
    print(possible_move[0])
    print(possible_move[1])


