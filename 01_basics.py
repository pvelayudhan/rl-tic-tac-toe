# Re-implementation of github.com/fcarsten/tic-tac-toe

from tictactoe import Game


outcomes = []

for i in range(100000):
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

# 12.5%
