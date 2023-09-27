from tictactoe import Game
import random
import numpy as np
from copy import deepcopy

def min_max(game):
    if game.done:
        if game.winner == None: # game was a draw
            return 0
        elif game.winner == 1: # player 1 wins
            return 1
        elif game.winner == 2: # player 2 wins
            return -1
    else:
        if game.turn % 2 == 0: # it's player 1's turn
            print("it's X turn")
        else: # it's player 2's turn
            print("it's O turn")


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
#game.play(2, 0)
#game.play(2, 2)
# game.play(2, 1)

game.board.show()

min_max(game)

game.board.squares

possible_moves = np.where(game.board.squares == 0)

possible_moves

move_values = list()

for possible_move in possible_moves:
    print(possible_move)




    #possible_game = deepcopy(game)
    #possible_game.play(possible_move[0], possible_move[1])
    #move_values.append(min_max(possible_game))



