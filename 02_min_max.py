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


# Shlone v mma
game.play(0, 0)

game.board.show()

min_max_agent.play(game)

game.board.show()

game.play(2, 2)

game.board.show()

min_max_agent.play(game)

game.board.show()

game.play(2, 1)

game.board.show()

min_max_agent.play(game)

game.board.show()

game.play(1, 0)

game.board.show()

min_max_agent.play(game)

game.board.show()

game.board.squares
