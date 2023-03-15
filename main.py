import tictactoe.board as tb

# Initialize board
board = tb.Board()

# Display board
board.show()

# Play a move
board.place_token(1, 2, 2)

board.show()

# Information about a particular square
board.squares
