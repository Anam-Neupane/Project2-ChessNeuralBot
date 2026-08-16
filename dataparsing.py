import chess.pgn
import io 

# A sample PGN game (Scholar's Mate) 

pgn_text = """
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1950"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""

game = chess.pgn.read_game(io.StringIO(pgn_text))

print("White:", game.headers["White"])
print("Result:", game.headers["Result"])

board = game.board()
for move in game.mainline_moves():
    print(f"  {board.fullmove_number}{'.' if board.turn else '...'} {board.san(move)}")
    board.push(move)

print("Game over:", board.is_game_over())
print("Result:", board.result())