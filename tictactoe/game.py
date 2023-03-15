from tictactoe.board import Board


class Game:

    def __init__(self):
        self.board = Board()
        self.turn = 0
        self.done = False
        self.winner = None



