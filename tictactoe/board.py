import numpy as np


class Board:

    """
    Board:
    a3 b3 c3
    a2 b2 c2
    a1 b1 c1
    """
    def __init__(self):
        self.squares = np.zeros(shape=(3, 3))

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
        rank3 = ("2 | ") + \
            self.int_to_token(self.squares[2][0]) + "   " + \
            self.int_to_token(self.squares[2][1]) + "   " + \
            self.int_to_token(self.squares[2][2])
        rank2 = ("1 | ") + \
            self.int_to_token(self.squares[1][0]) + "   " + \
            self.int_to_token(self.squares[1][1]) + "   " + \
            self.int_to_token(self.squares[1][2])
        rank1 = ("0 | ") + \
            self.int_to_token(self.squares[0][0]) + "   " + \
            self.int_to_token(self.squares[0][1]) + "   " + \
            self.int_to_token(self.squares[0][2])
        print(rank3)
        print(rank2)
        print(rank1)
        print("    ---------")
        print("    0   1   2")
        print("\n")
