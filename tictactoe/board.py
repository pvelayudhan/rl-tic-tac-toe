class Board:

    """
    Board:
    a3 b3 c3
    a2 b2 c2
    a1 b1 c1
    """
    def __init__(self):
        self.squares = {
            "a1": ".",
            "b1": ".",
            "c1": ".",
            "a2": ".",
            "b2": ".",
            "c2": ".",
            "a3": ".",
            "b3": ".",
            "c3": "."
        }


    def play(self, location, token):
        self.squares[location] = token


    def show(self):
        print("")
        print(f"3 | {self.squares['a3']}   {self.squares['b3']}   {self.squares['c3']}")
        print(f"2 | {self.squares['a2']}   {self.squares['b2']}   {self.squares['c2']}")
        print(f"1 | {self.squares['a1']}   {self.squares['b1']}   {self.squares['c1']}")
        print("    ---------")
        print("    a   b   c")


# Initialize board
board = Board()

# Display board
board.show()

# Play a move
board.play("a3", "x")
board.show()

# Information about a particular square
board.squares["a2"]
