from tictactoe import Game
import numpy as np
import tensorflow as tf

# The Q-table
Q = {}

game = Game()

# Some information about what is in a game
vars(game)


game.board.show()

game.play(1, 1)

state = game.board.calculate_hash()

state

Q[(state, state)] = 3
