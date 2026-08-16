# parse pgn files into numpy arrays suitable for pytorch training.

import chess
import chess.pgn
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from board_utils import board_to_tensor, move_to_index

def parse_pgn_to_arrays(pgn_path: str, max_games: int = 50_000,
                        min_elo: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a PGN file and extract (board_tensor, move_index, game_result) triples.

    Boards are stored as uint8 (they only contain 0/1 values), which uses 4x less
    RAM than float32 - important for large datasets. They are cast back to float32
    inside ChessDataset.__getitem__ when training.
    """
    CHUNK_SIZE = 250_000  # positions per chunk (~272 MB as uint8)

    pos_chunks = []
    mov_chunks = []
    val_chunks = []

    pos_chunk = np.empty((CHUNK_SIZE, 17, 8, 8), dtype=np.uint8)
    mov_chunk = np.empty((CHUNK_SIZE,), dtype=np.int64)
    val_chunk = np.empty((CHUNK_SIZE,), dtype=np.float32)
    offset = 0

    games_loaded = 0
    games_skipped = 0
    positions_total = 0

    with open(pgn_path, encoding='utf-8', errors='ignore') as f:
        pbar = tqdm(total=max_games, desc="Parsing PGN games")

        while games_loaded < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            try:
                white_elo = int(game.headers.get('WhiteElo', '0'))
                black_elo = int(game.headers.get('BlackElo', '0'))
            except ValueError:
                games_skipped += 1
                continue

            if white_elo < min_elo or black_elo < min_elo:
                games_skipped += 1
                continue

            # skip games without a clear result
            result = game.headers.get('Result', "*")
            if result == '1-0':
                game_value = 1.0
            elif result == '0-1':
                game_value = -1.0
            elif result == '1/2-1/2':
                game_value = 0.0
            else:
                games_skipped += 1
                continue

            # Extract positions and labels
            board = game.board()
            game_positions = []
            game_moves = []

            for move in game.mainline_moves():
                game_positions.append(board_to_tensor(board))
                game_moves.append(move_to_index(move))
                board.push(move)

            # Skip very short games (likely aborted or openings only)
            if len(game_moves) < 10:
                games_skipped += 1
                continue

            # Copy this game into the current chunk, starting a new one if full.
            n_game = len(game_positions)
            if offset + n_game > CHUNK_SIZE:
                pos_chunks.append(pos_chunk[:offset])
                mov_chunks.append(mov_chunk[:offset])
                val_chunks.append(val_chunk[:offset])
                pos_chunk = np.empty((CHUNK_SIZE, 17, 8, 8), dtype=np.uint8)
                mov_chunk = np.empty((CHUNK_SIZE,), dtype=np.int64)
                val_chunk = np.empty((CHUNK_SIZE,), dtype=np.float32)
                offset = 0

            pos_chunk[offset:offset + n_game] = np.array(game_positions, dtype=np.uint8)
            mov_chunk[offset:offset + n_game] = np.array(game_moves, dtype=np.int64)
            # game_value is always from White's perspective (+1/-1/0)
            val_chunk[offset:offset + n_game] = game_value
            offset += n_game

            games_loaded += 1
            positions_total += n_game
            pbar.update(1)
            pbar.set_postfix({'positions': positions_total, 'skipped': games_skipped})

        pbar.close()

    if offset > 0:
        pos_chunks.append(pos_chunk[:offset])
        mov_chunks.append(mov_chunk[:offset])
        val_chunks.append(val_chunk[:offset])

    print(f"\nLoaded  {games_loaded:,} games,  {positions_total:,} positions")
    print(f"Skipped {games_skipped:,} games (low ELO or no result)")

    if not pos_chunks:
        return (
            np.empty((0, 17, 8, 8), dtype=np.uint8),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    positions = np.concatenate(pos_chunks) if len(pos_chunks) > 1 else pos_chunks[0]
    moves = np.concatenate(mov_chunks) if len(mov_chunks) > 1 else mov_chunks[0]
    values = np.concatenate(val_chunks) if len(val_chunks) > 1 else val_chunks[0]

    return positions, moves, values


def save_array(positions: np.ndarray, moves: np.ndarray, values: np.ndarray, output_dir: str = 'data/processed') -> None:
    """Save processed array to disk so we don't re-parse the PGN every training run."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f'{output_dir}/positions.npy', positions)
    np.save(f'{output_dir}/moves.npy', moves)
    np.save(f'{output_dir}/values.npy', values)
    print(f"Saved {len(positions):,} positions to {output_dir}/")

def load_arrays(data_dir: str = 'data/processed') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-saved arrays. Much faster than re-parsing PGN."""

    return (
        np.load(f'{data_dir}/positions.npy'),
        np.load(f'{data_dir}/moves.npy'),
        np.load(f'{data_dir}/values.npy'),
    )

class ChessDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping our chess positions.
    A Dataset is the interface PyTorch uses to feed data to the training loop.
    __len__ -> number of samples
    __getitem__ -> one (input, label) pair by index
    """

    def __init__(self, positions: np.ndarray, moves: np.ndarray, values: np.ndarray):

        # Boards are stored as uint8 (only 0/1) to save 4x the RAM of float32.
        # They're cast to float32 on-the-fly in __getitem__.
        self.positions = torch.tensor(positions, dtype=torch.uint8)
        self.moves = torch.tensor(moves, dtype=torch.long)
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.positions[idx].float(), self.moves[idx], self.values[idx]

    # To process data

if __name__ == "__main__":
    import sys
    pgn = sys.argv[1] if len(sys.argv) > 1 else "data/lichess_elite_2023-01.pgn"
    max_g = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000

    positions, moves, values = parse_pgn_to_arrays(pgn, max_games=max_g)
    save_array(positions, moves, values)
    print("\n Dataset ready")
