# Let's try to convert board position to neural network inputs, and move indices. 

import chess
import numpy as np
import torch 

# We have to map the channels 
# There are 17 channels: 12 piece planes + 1 turn + 4 castling rights
# channel 0-5 : White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
# channel 6-11: Black pieces (same order) 
# channel 12: 1.0 everywhere if White to move, 0.0 if Black to move
# channels 13-16: Castline (White-K, White-Q, Black-K, Black-Q) - all -1 or all -0


CHANNEL_MAP = {
    
    (chess.WHITE, chess.PAWN): 0, 
    (chess.WHITE, chess.KNIGHT): 1, 
    (chess.WHITE, chess.BISHOP) : 2, 
    (chess.WHITE, chess.ROOK): 3, 
    (chess.WHITE, chess.QUEEN) : 4, 
    (chess.WHITE, chess.KING) : 5, 
    (chess.BLACK, chess.PAWN) : 6, 
    (chess.BLACK, chess.KNIGHT) : 7, 
    (chess.BLACK, chess.BISHOP) : 8, 
    (chess.BLACK, chess.ROOK) : 9, 
    (chess.BLACK, chess.QUEEN) : 10, 
    (chess.BLACK, chess.KING) : 11, 
}

def board_to_tensor(board: chess.Board) -> np.ndarray: 
    """
    Convert a chess.Board into a (17, 8 ,8) float32 numpy array. 
    """
    
    tensor = np.zeros((17,8,8), dtype=np.float32)
    
    # - piece planes (channel 0-11) 
    for (color, piece_type), channel in CHANNEL_MAP.items(): 
        for square in board.pieces(piece_type, color): 
            
            rank = square // 8  # 0 = rank1 (bottom) , 7 = rank8 (top) 
            file = square % 8   # 0 = a-file, 7= h-file
            tensor[channel][rank][file] = 1.0
    
    # - side to move (channel 12) 
    if board.turn == chess.WHITE: 
        tensor[12, :, :] = 1.0 # all ones plane
    
    # - castling rights (channel 13-16) 
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): 
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): 
        tensor[16, :, :] = 1.0
        
    return tensor


def move_to_index(move: chess.Move) -> int: 
    """
    Encode a move as an integer index 0 to 4095.
    Encoding: from_square * 64 + to_square 
    (4096 = 64 * 64 possible from/to combinations)  

    Any move to the back rank is always to promoted to queen.
    """
    return move.from_square * 64 + move.to_square

def index_to_move_squares(index: int) -> tuple[int, int]: 
    """
    Decode an index back to (from_square, to_square) 
    Inverse of move_to_index
    """
    from_square = index // 64
    to_square = index % 64
    return from_square, to_square

def mask_illegal_moves(logits: torch.Tensor, board: chess.Board) -> torch.Tensor: 
    """
    Set the logits of all illegal moves to -1e9 so they get ~0 probabilitiy after softmax.
    
    """
    mask = torch.full_like(logits, -1e9) 
    for move in board.legal_moves: 
        idx = move.from_square * 64 + move.to_square
        mask[idx] = logits[idx] 
    return mask 

if __name__ == "__main__": 
    board = chess.Board()
    tensor = board_to_tensor(board) 
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"White pawn count: {tensor[0].sum()}")
    print(f"Black pawn count: {tensor[6].sum()}") 
    print(f"Side to move (1 = White): {tensor[12, 0, 0]}")
    
    # testing move 
    move = chess.Move(chess.E2, chess.E4) 
    idx = move_to_index(move) 
    print(f"e2e4 index : {idx}")
    fr, to = index_to_move_squares(idx) 
    print(f"Decoded: from ={chess.square_name(fr)} to = {chess.square_name(to)}")
    
    
    
    