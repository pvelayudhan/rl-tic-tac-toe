from tictactoe import Game
from min_max_agent import MinMaxAgent
import random
import numpy as np
from copy import deepcopy

game = Game()
min_max_agent = MinMaxAgent()


# A drawn game
game = Game()

game.play(0, 0)
game.play(0, 1)
game.play(1, 2)
game.play(1, 0)
game.play(1, 1)
game.play(0, 2)
game.play(2, 0)

game.board.show()

min_max_agent.play(game)

game.board.show()

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
