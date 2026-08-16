import sys, os 

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path 
from tqdm import tqdm 

from model import ChessNet
from dataset import ChessDataset, load_arrays


# Hyperparameters

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512    # How many positions per gradient step. Larger = more stable gradients. 
EPOCHS = 15         # Full passes 'through the dataset. More = better (up to overfitting)
LR = 3e-4           # Learning rate: step size for gradient descent. 3e-4 is a safe Adam default. 
DATA_DIR = 'data/processed'
CKPT_DIR = 'checkpoints'
VAL_RATIO = 0.1 # Hold out 10% of data for validation (never used in training) 

NUM_WORKERS = 0 # Set to 2 on Linux/Mac. Keep 0 on Windows to avoid multiprocessing issues. 

# data 

print("Loading data... ")
positions, moves, values = load_arrays(DATA_DIR)
dataset = ChessDataset(positions, moves, values)

# Random split int train/ validation
val_size = int(len(dataset) * VAL_RATIO) 
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# DataLoader batches the data and shuffles it each epoch 
# pin_memory=True speeds up GPU transfer; num_workers for background Loading
train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))

print(f"Train: {len(train_ds):,} positions | Val: {len(val_ds):,} positions")

model = ChessNet(num_res_blocks=6, channels=128).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# CosineAnnealingLR: smoothly reduces LR from Lr to ~0 over EPOCHS. 
# This helps the model converge to a better minimum - Large steps early, fine-tuning late. 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

#Loss function: 
# CrossEntropyLoss for policy: compares predicted distribution over 4096 moves to 
# the one correct move (sparse one-hot label). Equivalent to -log(p_correct_move). 
# MSEloss for value: mean squared error between predicted eval and game outcome. 

policy_criterion = nn.CrossEntropyLoss()
value_criterion = nn.MSELoss()

# Training Loop 
Path(CKPT_DIR).mkdir(exist_ok=True)
best_val_loss = float('inf') 

for epoch in range(1, EPOCHS + 1): 
    
    model.train() 
    train_policy_loss = 0.0 
    train_correct_top1 = train_correct_top5 = train_total = 0
    
    pbar = tqdm(train_loader, desc = f"Epoch {epoch} / {EPOCHS} [TRAIN]", leave=False)
    for boards, move_indices, game_values in pbar: 
        boards = boards.to(DEVICE) 
        move_indices = move_indices.to(DEVICE) 
        game_values = game_values.to(DEVICE) 
        
        # zero gradients - Pytorch accumulates graident by default, so we must 
        # reset them eaach step. Forgetting this causes training to diverge. 
        optimizer.zero_grad()
        
        # Forward pass: run the network 
        policy_logits, value_pred = model(boards) 
        
        # Compute Losses 
        p_loss = policy_criterion(policy_logits, move_indices) 
        v_loss = value_criterion(value_pred.squeeze(), game_values) 
        loss = p_loss + 0.01 * v_loss # weight value loss Lower (policy is primary signal) 
        
        # Backward pass: compute gradients via backpropagation (chain rule) 
        loss.backward() 
        
        # Gradient clipping: cap gradient magnitued to 1.0. 
        # "Exploding gradients" can send weights to infinity; clipping prevents this. 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        
        # Update weights: w = w - lr * gradient 
        optimizer.step()         
        
        # Metrics 
        # Top-1 : did we pick the exact move played? 
        # Top-5 : was the correct move in our top-5 predictions? 
        topk = policy_logits.topk(5, dim=1).indices
        train_correct_top1 += (topk[:, 0] == move_indices).sum().item() 
        train_correct_top5 += (topk == move_indices.unsqueeze(1)).any(dim=1).sum().item()
        train_total  += len(move_indices) 
        train_policy_loss += p_loss.item() 
        
        pbar.set_postfix({
            'p_loss' : f'{p_loss.item():.4f}', 
            'top1': f'{train_correct_top1/train_total:.3f}', 
        })
        
    model.eval() 
    val_policy_loss = val_correct = val_total = 0 
    
    with torch.no_grad(): 
        for boards, move_indices, game_values in val_loader: 
            boards, move_indices, game_values = ( 
                    boards.to(DEVICE), move_indices.to(DEVICE), game_values.to(DEVICE))
            policy_logits, value_pred = model(boards) 
            p_loss = policy_criterion(policy_logits, move_indices) 
            val_policy_loss += p_loss.item() 
            val_correct += (policy_logits.argmax(dim=1) == move_indices).sum().item()
            val_total += len(move_indices)
            
    avg_train_loss = train_policy_loss / len(train_loader) 
    avg_val_loss = val_policy_loss / len(val_loader) 
    val_acc = val_correct / val_total
    train_top1 = train_correct_top1 / train_total 
    train_top5 = train_correct_top5 / train_total
    
    print(f"\n Epoch {epoch:02d}:"
          f"train_top1 = {train_top1:.3f} train_top5={train_top5:.3f} "
          f"train_loss = {avg_train_loss:.4f} val_loss = {avg_val_loss:.4f} val_acc={val_acc:.3f} ") 
    
    scheduler.step() 
    
    ckpt_path = f'{CKPT_DIR}/chess_net_epoch{epoch:02d}.pt'
    torch.save({
        'epoch': epoch, 
        'model_state': model.state_dict(), 
        'optimizer_state': optimizer.state_dict(), 
        'val_loss': avg_val_loss,
    }, ckpt_path)
    
    if avg_val_loss < best_val_loss: 
        best_val_loss = avg_val_loss 
        torch.save(model.state_dict(), f'{CKPT_DIR}/best_model.pt')
        print(f" New best model ! val loss = {best_val_loss:.4f}")

print(f"\n Training complete. Best model: {CKPT_DIR}/best_model.pt")




