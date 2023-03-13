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
        rank3 = ("3 | ") + \
            str(self.squares['a3']) + "   " + \
            str(self.squares['b3']) + "   " + \
            str(self.squares['c3'])
        rank2 = ("2 | ") + \
            str(self.squares['a2']) + "   " + \
            str(self.squares['b2']) + "   " + \
            str(self.squares['c2'])
        rank1 = ("1 | ") + \
            str(self.squares['a1']) + "   " + \
            str(self.squares['b1']) + "   " + \
            str(self.squares['c1'])
        print(rank3)
        print(rank2)
        print(rank1)
        print("    ---------")
        print("    a   b   c")
        print("\n")
