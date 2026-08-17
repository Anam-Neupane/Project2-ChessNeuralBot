# Loads the trained network once and turns a chess.Board into a move. 

import os
import sys
from typing import Optional
import chess
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0,SRC_DIR)

from model import ChessNet
from board_utils import board_to_tensor, mask_illegal_moves, index_to_move_squares

def _model_path():
    candidates = []
    if getattr(sys, "frozen", False):
        # PyInstaller --onedir keeps data files in the _internal directory.
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "checkpoints", "best_model.pt"))
        candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(sys.executable)),
            "checkpoints", "best_model.pt"))
    else:
        candidates.append(os.path.join(ROOT, "checkpoints", "best_model.pt"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]

MODEL_PATH = _model_path()
if not os.path.isfile(MODEL_PATH):
    raise RuntimeError("Model not found at %s"% MODEL_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ChessNet(num_res_blocks=6, channels=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """ Decode a 0-4095 index back into a chess.Move (always promote to queen.)
    """
    
    from_square, to_square = index_to_move_squares(index) 
    move = chess.Move(from_square, to_square)
    
    piece = board.piece_at(from_square)
    if piece is not None and piece.piece_type == chess.PAWN: 
        if chess.square_rank(to_square) in (0,7): 
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
    return move

def pick_move(board: chess.Board) -> Optional[chess.Move]: 
    """ Return the model's best legal move for 'board', or None if game is over."""
    
    if board.is_game_over(): 
        return None 
    
    with torch.no_grad(): 
        # (17, 8, 8) -> (1, 17, 8 , 8) batch dimension
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(DEVICE)
        policy_logits, _ = model(tensor) # (1, 4096) and (1,1)
        
        # Zero out every move that is not legal in this position. 
        masked = mask_illegal_moves(policy_logits[0].cpu(), board)
        best_index = int(masked.argmax())
        
        move = index_to_move(best_index, board) 
        # The final safety check: never return an illegal move. 
        return move if move in board.legal_moves else None
    