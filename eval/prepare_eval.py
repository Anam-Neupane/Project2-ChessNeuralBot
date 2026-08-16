import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import hashlib
import json
import chess
import chess.pgn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from board_utils import board_to_tensor, move_to_index

CHUNK_SIZE = 100_000
TRAIN_POS_PATH = os.path.join(ROOT, 'data', 'processed', 'positions.npy')
OUTPUT_DIR = os.path.join(HERE, 'data', 'processed_eval')


def tensor_hashes(rows: np.ndarray) -> list[str]:
    """sha256 of every flattened (17,8,8) uint8 tensor -> exact model-input identity."""
    flat = rows.reshape(len(rows), -1)
    return [hashlib.sha256(r.tobytes()).hexdigest() for r in flat]


def build_training_hash_set(train_pos_path: str = TRAIN_POS_PATH) -> set[str]:
    print("Hashing training positions (exact tensor identity)...")
    train = np.load(train_pos_path, mmap_mode='r')  # uint8 (N, 17, 8, 8)
    n = len(train)
    hashes: set[str] = set()
    step = 100_000
    for i in range(0, n, step):
        block = np.asarray(train[i:i + step])
        hashes.update(tensor_hashes(block))
        del block
    print(f"Training position hashes: {n:,}")
    return hashes


def parse_eval_pgn(pgn_path: str, train_hashes: set[str],
                   max_games: int = 10_000_000, min_elo: int = 0,
                   output_dir: str = OUTPUT_DIR):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pos_f = open(f'{output_dir}/positions.bin', 'wb')
    mov_f = open(f'{output_dir}/moves.bin', 'wb')
    val_f = open(f'{output_dir}/values.bin', 'wb')
    elo_f = open(f'{output_dir}/mover_elo.bin', 'wb')

    pos_chunk = np.empty((CHUNK_SIZE, 17, 8, 8), dtype=np.uint8)
    mov_chunk = np.empty((CHUNK_SIZE,), dtype=np.int64)
    val_chunk = np.empty((CHUNK_SIZE,), dtype=np.float32)
    elo_chunk = np.empty((CHUNK_SIZE,), dtype=np.int64)
    offset = 0

    games_loaded = games_skipped = positions_total = dropped = saved = 0
    elo_min, elo_max = np.iinfo(np.int64).max, np.iinfo(np.int64).min

    def finalize() -> None:
        nonlocal pos_chunk, mov_chunk, val_chunk, elo_chunk, offset, dropped, saved, elo_min, elo_max
        if offset == 0:
            return
        hashes = tensor_hashes(pos_chunk[:offset])
        keep = np.array([h not in train_hashes for h in hashes], dtype=bool)
        dropped += int(len(keep) - keep.sum())
        n_keep = int(keep.sum())
        if n_keep > 0:
            pos_f.write(pos_chunk[:offset][keep].tobytes())
            mov_f.write(mov_chunk[:offset][keep].tobytes())
            val_f.write(val_chunk[:offset][keep].tobytes())
            elo_f.write(elo_chunk[:offset][keep].tobytes())
            elo_min = min(elo_min, int(elo_chunk[:offset][keep].min()))
            elo_max = max(elo_max, int(elo_chunk[:offset][keep].max()))
            saved += n_keep
        del hashes, keep
        pos_chunk = np.empty((CHUNK_SIZE, 17, 8, 8), dtype=np.uint8)
        mov_chunk = np.empty((CHUNK_SIZE,), dtype=np.int64)
        val_chunk = np.empty((CHUNK_SIZE,), dtype=np.float32)
        elo_chunk = np.empty((CHUNK_SIZE,), dtype=np.int64)
        offset = 0

    with open(pgn_path, encoding='utf-8', errors='ignore') as f:
        pbar = tqdm(total=max_games, desc="Parsing eval PGN")
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

            result = game.headers.get('Result', '*')
            if result == '1-0':
                game_value = 1.0
            elif result == '0-1':
                game_value = -1.0
            elif result == '1/2-1/2':
                game_value = 0.0
            else:
                games_skipped += 1
                continue

            board = game.board()
            game_positions = []
            game_moves = []
            game_elos = []

            for move in game.mainline_moves():
                game_positions.append(board_to_tensor(board))
                game_moves.append(move_to_index(move))
                # rating of the player actually to move
                game_elos.append(white_elo if board.turn == chess.WHITE else black_elo)
                board.push(move)

            if len(game_moves) < 10:
                games_skipped += 1
                continue

            n_game = len(game_moves)
            if offset + n_game > CHUNK_SIZE:
                finalize()

            pos_chunk[offset:offset + n_game] = np.array(game_positions, dtype=np.uint8)
            mov_chunk[offset:offset + n_game] = np.array(game_moves, dtype=np.int64)
            val_chunk[offset:offset + n_game] = game_value
            elo_chunk[offset:offset + n_game] = np.array(game_elos, dtype=np.int64)
            offset += n_game

            games_loaded += 1
            positions_total += n_game
            pbar.update(1)
            pbar.set_postfix({'positions': positions_total, 'skipped': games_skipped})

        pbar.close()

    finalize()
    pos_f.close()
    mov_f.close()
    val_f.close()
    elo_f.close()

    print(f"\nLoaded  {games_loaded:,} games,  {positions_total:,} raw positions")
    print(f"Skipped {games_skipped:,} games (no result or short)")
    print(f"Dropped {dropped:,} positions seen in training set (tensor dedup)")
    print(f"Saved   {saved:,} held-out positions")

    meta = {
        'n_positions': saved,
        'elo_min': elo_min if saved else 0,
        'elo_max': elo_max if saved else 0,
    }
    with open(f'{output_dir}/meta.json', 'w') as mf:
        json.dump(meta, mf)


def load_eval_arrays(data_dir: str = OUTPUT_DIR):
    with open(f'{data_dir}/meta.json') as mf:
        meta = json.load(mf)
    n = meta['n_positions']
    positions = np.memmap(f'{data_dir}/positions.bin', dtype=np.uint8, mode='r',
                          shape=(n, 17, 8, 8))
    moves = np.memmap(f'{data_dir}/moves.bin', dtype=np.int64, mode='r', shape=(n,))
    values = np.memmap(f'{data_dir}/values.bin', dtype=np.float32, mode='r', shape=(n,))
    mover_elo = np.memmap(f'{data_dir}/mover_elo.bin', dtype=np.int64, mode='r', shape=(n,))
    return positions, moves, values, mover_elo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parse + dedup a held-out PGN for evaluation.')
    parser.add_argument('pgn', nargs='?', default=os.path.join(HERE, 'data', 'lichess_elite_2024-06.pgn'))
    parser.add_argument('--max-games', type=int, default=10_000_000)
    parser.add_argument('--min-elo', type=int, default=0)
    parser.add_argument('--train-positions', default=TRAIN_POS_PATH)
    args = parser.parse_args()

    train_hashes = build_training_hash_set(args.train_positions)
    parse_eval_pgn(args.pgn, train_hashes, max_games=args.max_games,
                   min_elo=args.min_elo)
    print("\nEval dataset ready.")
