import tictactoe.board as tb

# Initialize board
board = tb.Board()


# Display board
board.show()

# Play a move
board.play("a3", "x")
board.show()

# Information about a particular square
board.squares["a2"]
