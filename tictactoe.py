import numpy as np
import random


class Board:
    def __init__(self):
        self.squares = np.zeros(shape=(3, 3), dtype="int")

    def place_token(self, rank, file, token):
        self.squares[rank][file] = token

    def int_to_token(self, integer):
        if integer == 0:
            return "."
        elif integer == 1:
            return "x"
        elif integer == 2:
            return "o"

    def show(self):
        rank0 = (
            ("0 | ")
            + self.int_to_token(self.squares[0][0])
            + "   "
            + self.int_to_token(self.squares[0][1])
            + "   "
            + self.int_to_token(self.squares[0][2])
        )
        rank1 = (
            ("1 | ")
            + self.int_to_token(self.squares[1][0])
            + "   "
            + self.int_to_token(self.squares[1][1])
            + "   "
            + self.int_to_token(self.squares[1][2])
        )
        rank2 = (
            ("2 | ")
            + self.int_to_token(self.squares[2][0])
            + "   "
            + self.int_to_token(self.squares[2][1])
            + "   "
            + self.int_to_token(self.squares[2][2])
        )
        print(rank0)
        print(rank1)
        print(rank2)
        print("    ---------")
        print("    0   1   2")
        print("\n")


class Game:
    def __init__(self):
        self.board = Board()
        self.squares = self.board.squares
        self.turn = 0
        self.done = False
        self.p1_reward = 0
        self.winner = None

    def check_winner(self):
        # Because the board is represented by a 3x3 np.array and because the
        #  squares can have values of 0 (empty), 1 (player 1 token = x), and 2
        #  (player 2 token = o), the game is won when a row, column, or
        #  diagonal has a product of 1x1x1 = 1 or 2x2x2 = 8. The game is over
        #  when the whole array has a non-zero product.
        product_list = [
            np.prod(self.board.squares[:, 0]),
            np.prod(self.board.squares[:, 1]),
            np.prod(self.board.squares[:, 2]),
            np.prod(self.board.squares[0, :]),
            np.prod(self.board.squares[1, :]),
            np.prod(self.board.squares[2, :]),
            np.prod(self.board.squares.diagonal()),
            np.prod(np.fliplr(self.board.squares).diagonal()),
        ]
        if 1 in product_list:
            self.winner = 1
            self.p1_reward = 1
            self.done = True
        elif 8 in product_list:
            self.winner = 2
            self.p1_reward = 0
            self.done = True
        elif np.prod(self.board.squares) > 0:
            self.p1_reward = 0.5
            self.done = True

    def play(self, rank, file):
        # Ensure that the game isn't done
        if self.done is True:
            # print("No more moves can be played. The game is over!")
            return None
        # Ensure that the requested move is on the board
        if rank not in [0, 1, 2] or file not in [0, 1, 2]:
            # print("Illegal move. That's off the board!")
            return None
        # Ensure that the requested move is on an empty square
        if self.board.squares[rank][file] > 0:
            # print("Illegal move. That square is occupied!")
            return None
        # Assign token based on whether it is the turn of player 1 or player 2
        if self.turn % 2 == 0:
            token = 1
        else:
            token = 2
        # Play the move
        self.board.place_token(rank, file, token)
        # Refresh 'done' state and winner values
        self.check_winner()
        # Update the turn
        self.turn = self.turn + 1
        return None
