import chess.pgn

with open("data/lichess_elite_2023-01.pgn", encoding='utf-8', errors='ignore') as f: 
    for i in range(5): 
        game = chess.pgn.read_game(f)
        if game is None: 
            break 
        headers = game.headers
        move_count = len(list(game.mainline_moves()))
        print(f"Game {i+1}:") 
        print(f" {headers.get('White', '?')} ({headers.get('BlackElo', '?')})")
        print(f" Result: {headers.get('Result', '?')} | Moves: {move_count}")
        print(f" Time: {headers.get('TimeControl', '?')}")
        print()