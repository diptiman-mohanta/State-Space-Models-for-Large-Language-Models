import os
import math
import random
import argparse
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Dataset (unchanged)
# ------------------------
class CharDataset(Dataset):
    def __init__(self, lines: List[str], seq_len: int, char2idx: dict):
        self.seq_len = seq_len
        self.char2idx = char2idx
        self.idx2char = {v:k for k,v in char2idx.items()}
        text = "".join(lines)
        data = [char2idx.get(ch, char2idx["<unk>"]) for ch in text]
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start+self.seq_len]
        y = self.data[start+1:start+1+self.seq_len]
        return x, y

def build_vocab_from_lines(lines: List[str], extra_tokens: List[str] = None):
    if extra_tokens is None:
        extra_tokens = []
    text = "".join(lines)
    chars = sorted(list(set(text)))
    vocab = ["<pad>", "<unk>"] + extra_tokens + chars
    char2idx = {ch:i for i,ch in enumerate(vocab)}
    return char2idx

def load_and_preprocess_text(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                _, text = parts
            else:
                text = parts[0]
            lines.append(text)
    return lines

# ------------------------
# DiagSSMLayer (unchanged)
# ------------------------
class DiagSSMLayer(nn.Module):
    def __init__(self, dim_hidden: int, input_dim: int):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.in_proj = nn.Linear(input_dim, dim_hidden, bias=False)
        self.a = nn.Parameter(torch.full((dim_hidden,), 0.9))
        self.b = nn.Parameter(torch.randn(dim_hidden) * 0.1)
        self.c = nn.Parameter(torch.randn(dim_hidden) * 0.1)
        self.out_bias = nn.Parameter(torch.zeros(dim_hidden))

    def forward(self, u, x0=None):
        B, T, _ = u.shape
        h_in = self.in_proj(u)
        device = u.device
        if x0 is None:
            x = torch.zeros(B, self.dim_hidden, device=device)
        else:
            x = x0
        a = self.a.unsqueeze(0)
        b = self.b.unsqueeze(0)
        c = self.c.unsqueeze(0)
        ys = []
        for t in range(T):
            ut = h_in[:, t, :]
            x = a * x + b * ut
            y = c * x + self.out_bias
            ys.append(y.unsqueeze(1))
        y = torch.cat(ys, dim=1)
        return y, x

# ------------------------
# PureSSMCharLM
# ------------------------
class PureSSMCharLM(nn.Module):
    def __init__(self, vocab_size:int, emb_dim:int=128, ssm_hidden:int=512,
                 n_layers:int=6, seq_nonlin:bool=True, dropout:float=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.ssm_hidden = ssm_hidden
        
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        
        self.input_proj = nn.Linear(emb_dim, ssm_hidden)
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(n_layers):
            layer = DiagSSMLayer(dim_hidden=ssm_hidden, input_dim=ssm_hidden)
            self.layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(ssm_hidden))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.seq_nonlin = seq_nonlin
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(ssm_hidden, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        
        x = self.emb(idx)
        x = self.emb_dropout(x)
        residual = self.input_proj(x)
        
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.layer_norms, self.dropouts)):
            y, _ = layer(residual)
            y = norm(y)
            
            if self.seq_nonlin:
                y = torch.nn.functional.gelu(y)
            y = dropout(y)
            
            residual = residual + y
        
        logits = self.head(self.head_dropout(residual))
        return logits

# ------------------------
# Evaluation functions with accuracy
# ------------------------
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    loss_f = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.inference_mode():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            B, T, V = logits.shape
            logits_flat = logits.view(B*T, V)
            targets = yb.view(B*T)
            
            loss = loss_f(logits_flat, targets)
            total_loss += loss.item()
            
            predictions = logits_flat.argmax(dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_tokens += (B*T)
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    accuracy = correct_predictions / total_tokens
    return avg_loss, ppl, accuracy

def evaluate_ensemble(models, dataloader, device):
    for model in models:
        model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    loss_f = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.inference_mode():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            all_logits = [model(xb) for model in models]
            avg_logits = torch.stack(all_logits).mean(dim=0)
            
            B, T, V = avg_logits.shape
            logits_flat = avg_logits.view(B*T, V)
            targets = yb.view(B*T)
            
            loss = loss_f(logits_flat, targets)
            total_loss += loss.item()
            
            predictions = logits_flat.argmax(dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_tokens += (B*T)
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    accuracy = correct_predictions / total_tokens
    return avg_loss, ppl, accuracy

# ------------------------
# Averaged Model Class
# ------------------------
class AveragedSSMModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, idx):
        all_logits = [model(idx) for model in self.models]
        avg_logits = torch.stack(all_logits).mean(dim=0)
        return avg_logits

# ------------------------
# Plotting functions
# ------------------------
def plot_training_curves(history, ckpt_dir):
    epochs = history['epochs']
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    ax.plot(epochs, history['val_acc'], label='Val Accuracy', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Perplexity curves
    ax = axes[1, 0]
    ax.plot(epochs, history['train_ppl'], label='Train PPL', marker='o', markersize=3)
    ax.plot(epochs, history['val_ppl'], label='Val PPL', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Training and Validation Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation PPL only (for detailed view)
    ax = axes[1, 1]
    ax.plot(epochs, history['val_ppl'], label='Val PPL', marker='s', markersize=3, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Validation Perplexity (Detailed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {os.path.join(ckpt_dir, 'training_curves.png')}")

# ------------------------
# Training loop
# ------------------------
def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_lines = load_and_preprocess_text(args.train_text)
    val_lines = load_and_preprocess_text(args.val_text)
    test_lines = load_and_preprocess_text(args.test_text)
    
    char2idx = build_vocab_from_lines(train_lines + val_lines + test_lines)
    vocab_size = len(char2idx)
    print("Vocab size:", vocab_size)
    
    # Save vocab
    os.makedirs(args.ckpt_dir, exist_ok=True)
    vocab_file = os.path.join(args.ckpt_dir, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for ch, idx in sorted(char2idx.items(), key=lambda x: x[1]):
            f.write(f"{ch}\n")
    print(f"Saved vocabulary to {vocab_file}")

    # Create datasets and loaders
    train_ds = CharDataset(train_lines, args.seq_len, char2idx)
    val_ds = CharDataset(val_lines, args.seq_len, char2idx)
    test_ds = CharDataset(test_lines, args.seq_len, char2idx)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = PureSSMCharLM(vocab_size=vocab_size,
                          emb_dim=args.emb_dim,
                          ssm_hidden=args.ssm_hidden,
                          n_layers=args.n_layers,
                          dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_f = nn.CrossEntropyLoss()
    
    best_models = []  # List of (val_ppl, val_acc, epoch, model_path)
    
    # Training history for plotting
    history = {
        'epochs': [],
        'train_loss': [],
        'train_ppl': [],
        'train_acc': [],
        'val_loss': [],
        'val_ppl': [],
        'val_acc': []
    }

    # Training log
    log_file = os.path.join(args.ckpt_dir, "training_log.txt")
    with open(log_file, "w") as logf:
        logf.write("epoch,train_loss,train_ppl,train_acc,val_loss,val_ppl,val_acc\n")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        tokens = 0
        correct_predictions = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            B, T, V = logits.shape
            
            logits_flat = logits.view(B * T, V)
            targets = yb.view(B * T)
            
            loss = loss_f(logits_flat, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * (B * T)
            predictions = logits_flat.argmax(dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            tokens += (B * T)
            
            # Clear cache periodically
            if tokens % 10000 == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        train_loss = total_loss / tokens
        train_ppl = math.exp(train_loss)
        train_acc = correct_predictions / tokens
        
        val_loss, val_ppl, val_acc = evaluate(model, val_loader, device)
        
        # Store history
        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} ppl {train_ppl:.2f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_ppl {val_ppl:.2f} val_acc {val_acc:.4f}")

        # Log
        with open(log_file, "a") as logf:
            logf.write(f"{epoch},{train_loss},{train_ppl},{train_acc},{val_loss},{val_ppl},{val_acc}\n")

        # Save best 5 models based on validation PPL
        model_entry = (val_ppl, val_acc, epoch, None)  # path will be added after saving
        
        if len(best_models) < 5 or val_ppl < max(best_models, key=lambda x: x[0])[0]:
            # Save model with epoch and validation accuracy in filename
            model_filename = f"epoch_{epoch:03d}_valppl_{val_ppl:.2f}_valacc_{val_acc:.4f}.pt"
            model_path = os.path.join(args.ckpt_dir, model_filename)
            
            ck = {
                'model': model.state_dict(),
                'char2idx': char2idx,
                'args': vars(args),
                'epoch': epoch,
                'val_ppl': val_ppl,
                'val_acc': val_acc,
                'train_ppl': train_ppl,
                'train_acc': train_acc
            }
            torch.save(ck, model_path)
            
            model_entry = (val_ppl, val_acc, epoch, model_path)
            best_models.append(model_entry)
            best_models = sorted(best_models, key=lambda x: x[0])[:5]  # Keep top 5
            
            print(f"✓ Saved checkpoint: {model_filename}")
            
            # Clean up old models not in top 5
            all_checkpoints = [f for f in os.listdir(args.ckpt_dir) if f.startswith('epoch_') and f.endswith('.pt')]
            best_filenames = [os.path.basename(m[3]) for m in best_models]
            for ckpt_file in all_checkpoints:
                if ckpt_file not in best_filenames:
                    try:
                        os.remove(os.path.join(args.ckpt_dir, ckpt_file))
                    except:
                        pass

    print(f"\nTraining completed. Best 5 models saved. Logs: {log_file}")
    
    # Plot training curves
    plot_training_curves(history, args.ckpt_dir)
    
    # Print best 5 models
    print("\n=== BEST 5 MODELS ===")
    for i, (val_ppl, val_acc, epoch, path) in enumerate(best_models, 1):
        print(f"{i}. Epoch {epoch}: val_ppl={val_ppl:.2f}, val_acc={val_acc:.4f}")

    # Final test evaluation
    print("\n=== FINAL TEST EVALUATION ===")
    ensemble_models = []
    
    for i, (val_ppl, val_acc, epoch, model_path) in enumerate(best_models, 1):
        print(f"\nLoading model {i} (Epoch {epoch}): {os.path.basename(model_path)}")
        ck = torch.load(model_path, map_location=device)
        test_model = PureSSMCharLM(
            vocab_size=vocab_size,
            emb_dim=args.emb_dim,
            ssm_hidden=args.ssm_hidden,
            n_layers=args.n_layers,
            dropout=args.dropout
        ).to(device)
        test_model.load_state_dict(ck['model'])
        ensemble_models.append(test_model)
        
        test_loss, test_ppl, test_acc = evaluate(test_model, test_loader, device)
        print(f"  Test PPL: {test_ppl:.2f}, Test Accuracy: {test_acc:.4f}")

    # Evaluate ensemble
    print("\n=== ENSEMBLE EVALUATION (Average of Best 5 Models) ===")
    ensemble_loss, ensemble_ppl, ensemble_acc = evaluate_ensemble(ensemble_models, test_loader, device)
    print(f"Ensemble Test PPL: {ensemble_ppl:.2f}")
    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    
    # Create and save averaged model
    print("\n=== Creating Averaged Model ===")
    avg_model = PureSSMCharLM(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        ssm_hidden=args.ssm_hidden,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    # Average the parameters
    avg_state_dict = {}
    with torch.no_grad():
        for key in ensemble_models[0].state_dict().keys():
            avg_state_dict[key] = torch.stack([m.state_dict()[key].float() for m in ensemble_models]).mean(0)
    
    avg_model.load_state_dict(avg_state_dict)
    
    # Save averaged model
    avg_model_path = os.path.join(args.ckpt_dir, "averaged_best5_model.pt")
    torch.save({
        'model': avg_model.state_dict(),
        'char2idx': char2idx,
        'args': vars(args),
        'source_epochs': [epoch for _, _, epoch, _ in best_models]
    }, avg_model_path)
    print(f"Saved averaged model to {avg_model_path}")
    
    # Evaluate averaged model
    avg_test_loss, avg_test_ppl, avg_test_acc = evaluate(avg_model, test_loader, device)
    print(f"\nAveraged Model Test PPL: {avg_test_ppl:.2f}")
    print(f"Averaged Model Test Accuracy: {avg_test_acc:.4f}")
    
    # Save final results
    results_file = os.path.join(args.ckpt_dir, "final_test_results.txt")
    with open(results_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("FINAL TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Individual Model Test Results:\n")
        f.write("-"*60 + "\n")
        for i, (val_ppl, val_acc, epoch, path) in enumerate(best_models, 1):
            ck = torch.load(path, map_location='cpu')
            f.write(f"Model {i} (Epoch {epoch}):\n")
            f.write(f"  File: {os.path.basename(path)}\n")
            f.write(f"  Validation PPL: {val_ppl:.2f}, Validation Acc: {val_acc:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ENSEMBLE RESULTS (Inference-time averaging):\n")
        f.write("-"*60 + "\n")
        f.write(f"Ensemble Test PPL: {ensemble_ppl:.2f}\n")
        f.write(f"Ensemble Test Accuracy: {ensemble_acc:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("AVERAGED MODEL RESULTS (Parameter averaging):\n")
        f.write("-"*60 + "\n")
        f.write(f"Averaged Model Test PPL: {avg_test_ppl:.2f}\n")
        f.write(f"Averaged Model Test Accuracy: {avg_test_acc:.4f}\n")
    
    print(f"\nAll results saved to {results_file}")
    print(f"Training curves saved to {os.path.join(args.ckpt_dir, 'training_curves.png')}")

# ------------------------
# CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_text", type=str, required=True)
    p.add_argument("--val_text", type=str, required=True)
    p.add_argument("--test_text", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--ssm_hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints_enhanced")
    return p.parse_args()

def main():
    args = parse_args()
    train_loop(args)

if __name__ == "__main__":
    main()