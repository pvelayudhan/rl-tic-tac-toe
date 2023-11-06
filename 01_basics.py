#!/usr/bin/env python3

"""Re-implemention of Part 1 of https://github.com/fcarsten/tic-tac-toe

This script evaluates the percentage of tic-tac-toe player 1 (X) wins, player 2
(O) wins, and draws between two agents playing moves randomly.

"""

from tictactoe import Game

outcomes = []

for i in range(1000):
    print(i)
    game = Game()
    while not game.done:
        game.play_random_move()
    outcomes.append(game.winner)

n_draws = outcomes.count(None)
n_p1_win = outcomes.count(1)
n_p2_win = outcomes.count(2)
print(n_draws / len(outcomes))
print(n_p1_win / len(outcomes))
print(n_p2_win / len(outcomes))

"""
12.5% draw
58.6% P1 win
28.9% P2 win
"""
