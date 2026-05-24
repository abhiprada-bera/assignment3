"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import collections

from model import Transformer
from dataset import Multi30kDataset
from utils import make_src_mask, make_tgt_mask, LabelSmoothingLoss, NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    # Get pad index dynamically
    pad_idx = model.src_vocab.stoi.get('<pad>', 1)

    pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{'Train' if is_train else 'Val'}]")
    
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)

        # Decoder inputs: target shifted right by one (<sos> ... <eos> -> <sos> ...)
        tgt_input = tgt[:, :-1]
        # Loss targets: target shifted left by one (<sos> ... <eos> -> ... <eos>)
        tgt_y = tgt[:, 1:]

        # Create source and target masks
        src_mask = make_src_mask(src, pad_idx=pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx)

        # Forward pass
        if is_train:
            logits = model(src, tgt_input, src_mask, tgt_mask)
        else:
            with torch.no_grad():
                logits = model(src, tgt_input, src_mask, tgt_mask)

        # Flatten logits and targets for computing cross-entropy loss
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        tgt_y_flat = tgt_y.contiguous().view(-1)

        loss = loss_fn(logits_flat, tgt_y_flat)

        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Track loss
        non_pad_tokens = (tgt_y != pad_idx).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens
        batch_count += 1

        pbar.set_postfix(loss=loss.item())

    return total_loss / max(1, total_tokens)


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3, # <eos>
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    pad_idx = model.tgt_vocab.stoi.get('<pad>', 1)

    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device) # [1, 1]

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        for _ in range(max_len - 1):
            tgt_len = ys.size(1)
            # Create padding and causal target mask
            tgt_pad_mask = (ys == pad_idx).unsqueeze(1).unsqueeze(2)
            subsequent_mask = torch.triu(torch.ones((1, tgt_len, tgt_len), device=device), diagonal=1).bool().unsqueeze(1)
            tgt_mask = tgt_pad_mask | subsequent_mask
            
            out = model.decode(memory, src_mask, ys, tgt_mask)
            prob = out[:, -1, :]
            next_word = prob.argmax(dim=-1).item()

            ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
            if next_word == end_symbol:
                break
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def compute_fallback_bleu(references: list[list[str]], hypotheses: list[str]) -> float:
    """
    A pure Python implementation of corpus BLEU-4 calculation as fallback
    when Perl is missing from the environment.
    """
    def get_ngrams(segment, max_order):
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    clipped_counts = [0, 0, 0, 0]
    total_counts = [0, 0, 0, 0]
    r_len = 0
    c_len = 0

    for refs, hyp in zip(references, hypotheses):
        hyp_tokens = hyp.split()
        c_len += len(hyp_tokens)
        
        # Reference token lists
        ref_tokens_list = [ref.split() for ref in refs]
        closest_ref_len = min([len(r) for r in ref_tokens_list], key=lambda x: abs(x - len(hyp_tokens)))
        r_len += closest_ref_len
        
        # Target ngrams
        hyp_ngrams = get_ngrams(hyp_tokens, 4)
        
        # Max reference counts
        ref_ngrams = collections.Counter()
        for ref_tokens in ref_tokens_list:
            for ngram, count in get_ngrams(ref_tokens, 4).items():
                ref_ngrams[ngram] = max(ref_ngrams[ngram], count)
                
        # Aggregate clipped counts
        for ngram, count in hyp_ngrams.items():
            order = len(ngram) - 1
            clipped_counts[order] += min(count, ref_ngrams.get(ngram, 0))
            total_counts[order] += count

    p = []
    for i in range(4):
        if total_counts[i] == 0:
            p.append(0.0)
        else:
            p.append(clipped_counts[i] / total_counts[i])

    if c_len == 0:
        bp = 0.0
    elif c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1 - r_len / c_len)

    log_precisions = []
    for p_val in p:
        if p_val > 0:
            log_precisions.append(math.log(p_val))
        else:
            log_precisions.append(-999.0)
            
    score = bp * math.exp(sum(log_precisions) / 4.0) * 100
    return score


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()
    references = []
    hypotheses = []

    start_symbol = model.tgt_vocab.stoi.get('<sos>', 2)
    end_symbol = model.tgt_vocab.stoi.get('<eos>', 3)
    pad_idx = model.src_vocab.stoi.get('<pad>', 1)

    print("Evaluating BLEU score on test set...")
    for src, tgt in tqdm(test_dataloader, desc="Decoding test samples"):
        src = src.to(device)
        for b in range(src.size(0)):
            src_seq = src[b:b+1] # [1, src_len]
            src_mask = (src_seq == pad_idx).unsqueeze(1).unsqueeze(2)

            # Greedy Decode
            ys = greedy_decode(model, src_seq, src_mask, max_len, start_symbol, end_symbol, device)

            # Convert hypothesis token indices to string
            hyp_tokens = []
            for idx in ys.squeeze(0).tolist():
                token = tgt_vocab.lookup_token(idx)
                if token not in ['<sos>', '<eos>', '<pad>']:
                    hyp_tokens.append(token)
            hyp_str = " ".join(hyp_tokens)
            hypotheses.append(hyp_str)

            # Convert reference token indices to string
            ref_tokens = []
            for idx in tgt[b].tolist():
                token = tgt_vocab.lookup_token(idx)
                if token not in ['<sos>', '<eos>', '<pad>']:
                    ref_tokens.append(token)
            ref_str = " ".join(ref_tokens)
            references.append([ref_str])

    # Try using the system-installed bleu package
    import bleu
    try:
        bleu_score = bleu.list_bleu(references, hypotheses)
        if bleu_score == -1:
            print("Perl-based BLEU returned -1 (likely missing Perl). Computing Python fallback BLEU...")
            bleu_score = compute_fallback_bleu(references, hypotheses)
    except Exception as e:
        print(f"Failed to run Perl-based BLEU ({e}). Computing Python fallback BLEU...")
        bleu_score = compute_fallback_bleu(references, hypotheses)

    print(f"Evaluation finished. BLEU score: {bleu_score:.4f}")
    return bleu_score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.
    """
    model_config = {
        'src_vocab_size': model.src_embed.num_embeddings,
        'tgt_vocab_size': model.tgt_embed.num_embeddings,
        'd_model': model.d_model,
        'N': len(model.encoder.layers),
        'num_heads': model.encoder.layers[0].self_attn.num_heads,
        'd_ff': model.encoder.layers[0].feed_forward.linear1.out_features,
        'dropout': model.encoder.layers[0].dropout1.p,
    }

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config': model_config
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved successfully to {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.
    """
    if os.path.exists(path):
        print(f"Restoring checkpoint from {path}...")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch']
    else:
        print(f"No checkpoint found at {path}")
        return -1


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def pad_collate_fn(batch, pad_idx=1):
    src_list, tgt_list = [], []
    for src, tgt in batch:
        src_list.append(torch.tensor(src))
        tgt_list.append(torch.tensor(tgt))
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    import os
    os.environ["WANDB_MODE"] = "offline"
    import wandb
    try:
        wandb.init(
            project="da6401-a3",
            config={
                "d_model": 128,
                "N": 2,
                "num_heads": 4,
                "d_ff": 256,
                "dropout": 0.1,
                "batch_size": 128,
                "num_epochs": 2,
                "warmup_steps": 4000,
                "learning_rate": 1.0,
                "label_smoothing": 0.1,
            }
        )
        config = wandb.config
        use_wandb = True
    except Exception as e:
        print("wandb initialization failed or was skipped. Logging metrics to standard output only:", e)
        class ConfigFallback:
            d_model = 128
            N = 2
            num_heads = 4
            d_ff = 256
            dropout = 0.1
            batch_size = 128
            num_epochs = 2
            warmup_steps = 4000
            learning_rate = 1.0
            label_smoothing = 0.1
        config = ConfigFallback()
        use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build datasets
    train_dataset = Multi30kDataset(split='train')
    val_dataset = Multi30kDataset(split='validation')
    test_dataset = Multi30kDataset(split='test')

    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab
    
    pad_idx = src_vocab.stoi.get('<pad>', 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_collate_fn(b, pad_idx)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate_fn(b, pad_idx)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate_fn(b, pad_idx)
    )

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.d_model,
        N=config.N,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        checkpoint_path=None
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Instantiate NoamScheduler from utils
    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps
    )

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=pad_idx,
        smoothing=config.label_smoothing
    )

    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        train_loss = run_epoch(
            data_iter=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=str(device)
        )
        
        val_loss = run_epoch(
            data_iter=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=str(device)
        )

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        if use_wandb:
            wandb.log(metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                path="checkpoint.pt"
            )

    load_checkpoint("checkpoint.pt", model)
    bleu_score = evaluate_bleu(model, test_loader, tgt_vocab, str(device))
    
    print(f"Final Test BLEU Score: {bleu_score:.4f}")
    
    if use_wandb:
        wandb.log({'test_bleu': bleu_score})
        wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
