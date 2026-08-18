# Neural Chess Engine

A neural network-based chess engine trained on grandmaster games, served over HTTP to play against the C++ raylib chess game or standalone.

## Overview

Chess_bot is a supervised-learning chess engine built with PyTorch. It trains a convolutional neural network (ChessNet) on Lichess Elite games (2400+ ELO) to predict strong moves and evaluate positions. A Flask server exposes the model over HTTP so the C++ chess game can request moves in real time.

### Expected Performance

I wanted to build a bot that could defeat me a 1500 rated player in Chess.com it was a wishfull thinking at first but i just had to try if it would be possible. I tried but using this method it is not quite possible from my access to resources that are T4 gpus that Google colab provides, It took me 3 hours to train this model alone. This was a good experience at least. The Neural model is decent at the openings, decent at middle game but is worst at endgame. It always promotes to the queen as intended and it will randomly exchange materials without the understanding of weights of it's exchange. 

### System Architecture

```
 C++ Raylib Game                Python Flask Server
 ChessGame                          Chess_bot
                                
 POST /get_move                 board_to_tensor() (17x8x8)
 {"moves":["e2e4",...]}  ───►   ChessNet (PyTorch CNN)
                                policy logits (4096)
 ◄──────────────────────  {"move":"g1f3"}
                                mask illegal moves
                                argmax -> best move
```

## How to Use

### 1. Training the Model

#### Download Training Data

```powershell
# Run from Chess_bot root
python scraptrainingdata.py
```

This downloads the Lichess Elite database (~150MB compressed) to `data/`.

#### Process PGN into Training Arrays

```powershell
python src/dataset.py data/lichess_elite_2023-01.pgn 50000
```

This parses PGN games into `data/processed/` containing `positions.npy`, `moves.npy`, and `values.npy`.

#### Train

```powershell
# Local training (slow on CPU)
python src/train.py
```

For faster training, use the Google Colab notebooks:

- **`colab_train.ipynb`** - Upload the processed data and train on a free GPU
- **`colab_eval.ipynb`** - Evaluate model checkpoints against held-out positions

Checkpoints are saved to `checkpoints/`.

### 2. Running the Server

```powershell
python server/main.py --host 127.0.0.1 --port 5000
```

The server loads the trained model and listens for move requests.

### 3. Testing the Server

```powershell
# Single move request (PowerShell)
Invoke-RestMethod -Uri http://127.0.0.1:5000/get_move -Method Post `
  -ContentType "application/json" -Body '{"moves": []}'

# Self-play test
python test_client.py
```

### 4. Integration with the C++ Game

The C++ game (`Project1-Chess`) uses `NeuralEngine` (in `src/engine/NeuralEngine.cpp`) which:

1. Spawns the Python server as a child process on startup
2. Polls `GET /health` until the server is ready
3. Sends `POST /get_move` with the game's UCI move list each turn
4. Parses the JSON response and converts the UCI move to screen coordinates

Toggle between Stockfish and Neural Bot from the engine setup menu in-game.

### 5. Building a Standalone Executable

```powershell
pip install -r requirements-build.txt
pyinstaller neural-bot.spec
```

The distributable is created in `dist/neural-bot/`.

## Tech Stack

| Component | Technology |
|---|---|
| Neural network | PyTorch 2.1 (CNN with residual blocks) |
| Chess logic | python-chess 1.10 |
| API server | Flask 3.0 + Waitress |
| Training data | Lichess Elite PGN (2400+ ELO) |
| Cloud training | Google Colab (free GPU) |
| Packaging | PyInstaller 6.12 |

## Project Structure

```text
Chess_bot/
|-- README.md
|-- requirements.txt                 (Python dependencies)
|-- requirements-build.txt           (PyInstaller for standalone build)
|-- neural-bot.spec                  (PyInstaller spec file)
|-- scraptrainingdata.py             (Download Lichess Elite PGN)
|-- dataparsing.py                   (PGN parsing example)
|-- explore_dataset.py               (Quick data inspection)
|-- test_client.py                   (HTTP self-play test)
|-- src/
|   |-- board_utils.py               (Board -> tensor, move encoding, illegal move masking)
|   |-- dataset.py                   (PGN parsing -> numpy arrays, PyTorch Dataset)
|   |-- model.py                     (ChessNet CNN architecture)
|   |-- train.py                     (Training loop with validation)
|   `-- search.py                    (Load model, pick best legal move)
|-- server/
|   |-- app.py                       (Flask HTTP server)
|   `-- main.py                      (Server entry point with CLI args)
|-- eval/
|   |-- eval_new.py                  (Evaluate checkpoints, estimate ELO)
|   |-- prepare_eval.py              (Parse held-out PGN with dedup)
|   |-- scrape_eval_data.py          (Download evaluation dataset)
|   `-- data/                        (Eval dataset, generated)
|-- checkpoints/
|   |-- best_model.pt                (Current best weights)
|   `-- best_model2.pt               (Previous best)
|-- data/
|   |-- lichess_elite_2023-01.pgn    (Training PGN, gitignored)
|   `-- processed/                   (Numpy arrays, gitignored)
|-- colab_train.ipynb                (Google Colab training notebook)
|-- colab_eval.ipynb                 (Google Colab evaluation notebook)
```

## Neural Network Architecture

The model (`ChessNet`) is a simplified AlphaZero-style CNN:

- **Input**: `(batch, 17, 8, 8)` - 17-channel board representation
  - Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
  - Channels 6-11: Black pieces (same order)
  - Channel 12: Side to move (all 1s if White, all 0s if Black)
  - Channels 13-16: Castling rights (Kingside/Queenside for each side)
- **Architecture**: Stem convolution -> 6 residual blocks (128 channels) -> two heads
- **Policy head**: Predicts the best move from 4096 possible from-square/to-square combinations
- **Value head**: Predicts position evaluation from -1 (Black winning) to +1 (White winning)
- **Parameters**: ~4.6M

### Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=3e-4) |
| Scheduler | CosineAnnealing |
| Batch size | 512 |
| Epochs | 15 |
| Validation split | 10% |
| Loss | CrossEntropy (policy) + 0.01 * MSE (value) |
| Gradient clipping | max norm = 1.0 |

### Move Encoding

Moves are encoded as a single integer `0-4095`: `from_square * 64 + to_square`. During inference, all illegal moves are masked to `-1e9` so the argmax always selects a legal move.

## Evaluation

The `eval/` directory contains tools to measure model strength:

1. **`scrape_eval_data.py`** - Download a held-out month of Lichess Elite games
2. **`prepare_eval.py`** - Parse and deduplicate against the training set (exact tensor-level dedup via SHA-256)
3. **`eval_new.py`** - Evaluate checkpoints, compute top-1/top-5 accuracy per ELO bin, and estimate the model's playing strength

Results are saved to `eval/data/processed_eval/results.json`.

## Prerequisites

### Python

```powershell
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch is ~2GB. For CPU-only: `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu`

### Verify Installation

```python
python -c "import chess, torch, flask; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Training on Google Colab

1. Open `colab_train.ipynb` in Google Colab
2. Run all cells - training uses the free GPU tier
3. Download the resulting `checkpoints/best_model.pt`
4. Place it in the local `checkpoints/` folder

