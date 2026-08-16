import requests
import chess

SERVER = "http://127.0.0.1:5000"

def best_move(moves):
    r = requests.post(f"{SERVER}/get_move", json={"moves":moves}, timeout=30)
    
    r.raise_for_status()
    return r.json()["move"]

board = chess.Board()
history = []
for ply in range(160):
    if board.is_game_over():
        print("\nGame over:", board.result())
        break
    uci = best_move(history)
    print(uci, end= " ")
    if ply % 2 == 1: 
        print()
    board.push_uci(uci)
    history.append(uci) 
    