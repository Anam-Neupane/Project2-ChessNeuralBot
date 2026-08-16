import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)

import json
import torch
import torch.nn as nn
import numpy as np

from model import ChessNet
from prepare_eval import load_eval_arrays

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512
DATA_DIR = os.path.join(HERE, 'data', 'processed_eval')
CKPTS = [
    os.path.join(ROOT, 'checkpoints', 'best_model.pt'),
    os.path.join(ROOT, 'checkpoints', 'best_model2.pt'),
    os.path.join(ROOT, 'checkpoints', 'best_model3.pt'),
]
# mover-rating bins (last edge is sentinel for "and up")
ELO_EDGES = [2300, 2400, 2500, 2600, 2800]
BIN_CENTERS = [2350, 2450, 2550, 2700, 2850]
BIN_LABELS = ['2300-2400', '2400-2500', '2500-2600', '2600-2800', '2800+']
TARGET_MATCH = 0.5
MAX_PER_BIN = 200_000  # cap positions per ELO bin (CPU eval time)


def estimate_elo(top1: np.ndarray, target: float = TARGET_MATCH):
    """Rating where top-1 match rate crosses `target`, by linear interpolation."""
    valid = ~np.isnan(top1)
    centers = np.array(BIN_CENTERS, dtype=float)[valid]
    vals = top1[valid]
    if len(vals) == 0:
        return None
    for i in range(len(vals) - 1):
        hi, lo = vals[i], vals[i + 1]
        if (hi - target) * (lo - target) <= 0 and hi != lo:
            t = (target - lo) / (hi - lo)
            return centers[i] + t * (centers[i + 1] - centers[i])
    if vals[0] < target:
        return f"< {centers[0]}"
    if vals[-1] > target:
        return f"> {centers[-1]}"
    return None


def main() -> None:
    positions, moves, values, mover_elo = load_eval_arrays(DATA_DIR)
    n = len(moves)
    bin_idx = np.digitize(mover_elo, ELO_EDGES)  # 0..len(EDGES) => 5 bins

    # deterministic per-bin cap to bound CPU eval time
    sel_ids = []
    all_ids = np.arange(n)
    for b in range(len(BIN_LABELS)):
        ids = all_ids[bin_idx == b]
        if len(ids) > MAX_PER_BIN:
            ids = ids[np.linspace(0, len(ids) - 1, MAX_PER_BIN, dtype=int)]
        sel_ids.append(ids)
    sel_ids = np.concatenate(sel_ids)
    sel_ids = np.sort(sel_ids)

    print(f"Eval set: {n:,} positions (using {len(sel_ids):,} after {MAX_PER_BIN:,}/bin cap)"
          f" | Device: {DEVICE}")
    print(f"Mover ELO range: [{mover_elo.min()}, {mover_elo.max()}]\n")

    policy_criterion = nn.CrossEntropyLoss()
    results = {}

    for ckpt in CKPTS:
        if not os.path.exists(ckpt):
            print(f"SKIP (missing): {ckpt}\n")
            continue

        model = ChessNet(num_res_blocks=6, channels=128).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        n_sel = len(sel_ids)
        preds_top5 = np.empty((n_sel, 5), dtype=np.int64)
        true_moves = np.empty((n_sel,), dtype=np.int64)
        p_loss_sum = 0.0
        total = 0

        with torch.no_grad():
            for start in range(0, n_sel, BATCH_SIZE):
                ids = sel_ids[start:start + BATCH_SIZE]
                boards = torch.from_numpy(np.asarray(positions[ids])).float().to(DEVICE)
                move_indices = torch.from_numpy(np.asarray(moves[ids])).to(DEVICE)

                policy_logits, _ = model(boards)

                p_loss_sum += policy_criterion(policy_logits, move_indices).item() * len(move_indices)
                topk = policy_logits.topk(5, dim=1).indices.cpu().numpy()
                m = len(move_indices)
                preds_top5[total:total + m] = topk
                true_moves[total:total + m] = move_indices.cpu().numpy()
                total += m

        # per ELO bin
        bin_top1 = np.full(len(BIN_LABELS), np.nan)
        bin_top5 = np.full(len(BIN_LABELS), np.nan)
        bin_counts = np.zeros(len(BIN_LABELS), dtype=int)
        for b in range(len(BIN_LABELS)):
            mask = bin_idx[sel_ids] == b
            if mask.sum() == 0:
                continue
            bin_counts[b] = mask.sum()
            bin_top1[b] = (preds_top5[mask, 0] == true_moves[mask]).mean()
            bin_top5[b] = (preds_top5[mask] == true_moves[mask, None]).any(axis=1).mean()

        top1_all = (preds_top5[:, 0] == true_moves).mean()
        top5_all = (preds_top5 == true_moves[:, None]).any(axis=1).mean()
        p_loss = p_loss_sum / total

        elo_est = estimate_elo(bin_top1)

        results[os.path.basename(ckpt)] = {
            'top1': top1_all, 'top5': top5_all, 'p_loss': p_loss,
            'bin_counts': bin_counts, 'bin_top1': bin_top1, 'bin_top5': bin_top5,
            'elo_est': elo_est,
        }

        print(f"=== {ckpt} ===")
        print(f"  aggregate top-1 : {top1_all:.3f}")
        print(f"  aggregate top-5 : {top5_all:.3f}")
        print(f"  policy loss     : {p_loss:.4f}")
        print(f"  match rate by mover ELO:")
        print(f"    {'band':<12}{'positions':>12}{'top-1':>9}{'top-5':>9}")
        for b in range(len(BIN_LABELS)):
            if bin_counts[b] == 0:
                print(f"    {BIN_LABELS[b]:<12}{'0':>12}{'-':>9}{'-':>9}")
            else:
                print(f"    {BIN_LABELS[b]:<12}{bin_counts[b]:>12,}{bin_top1[b]:>9.3f}{bin_top5[b]:>9.3f}")
        print(f"  estimated strength : {elo_est}\n")

        del preds_top5, true_moves

    if results:
        print("Comparison (higher better for top-1/top-5, lower for p_loss):")
        for metric in ['top1', 'top5', 'p_loss']:
            best = max(results, key=lambda k: results[k][metric] if metric != 'p_loss' else -results[k][metric])
            print(f"  {metric:6s} -> {best}")

        results_path = os.path.join(DATA_DIR, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda v: v.tolist())
        print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
