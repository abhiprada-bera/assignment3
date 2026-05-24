"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import masking helpers from utils for restructuring requirements
from utils import make_src_mask, make_tgt_mask

# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
    scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -1e9 before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scaling = math.sqrt(d_k) if scale else 1.0
    # scores: [..., seq_q, seq_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scaling
    
    if mask is not None:
        scores = scores.masked_fill(mask == True, -1e9)
        
    attn_w = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        attn_w = dropout(attn_w)
        
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, scale: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.scale     = scale

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        batch = query.size(0)

        # 1. Project and split into heads
        # Shape change: [batch, seq, d_model] -> [batch, seq, num_heads, d_k] -> [batch, num_heads, seq, d_k]
        q = self.q_proj(query).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Compute attention
        x, _ = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout, scale=self.scale)

        # 3. Concat and apply final output projection
        # Shape change: [batch, num_heads, seq_q, d_k] -> [batch, seq_q, num_heads, d_k] -> [batch, seq_q, d_model]
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.out_proj(x)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute Positional Encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]
 
        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  
        """
        # x is [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, scale: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, scale=scale)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Self-Attention + Residual & LayerNorm (Post-LN)
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN + Residual & LayerNorm (Post-LN)
        ffn_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, scale: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, scale=scale)
        self.src_attn = MultiHeadAttention(d_model, num_heads, dropout, scale=scale)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # 1. Masked Self-Attention + Residual & LayerNorm
        attn_out1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out1))
        
        # 2. Cross-Attention + Residual & LayerNorm
        attn_out2 = self.src_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(attn_out2))
        
        # 3. Feed-Forward + Residual & LayerNorm
        ffn_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int = 7853,
        tgt_vocab_size: int = 5893,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = "checkpoint.pt",
        google_drive_id: str = None, # Can be set dynamically
    ) -> None:
        super().__init__()
        
        # Download checkpoint file using gdown if we don't have it locally and drive ID is provided
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path) and google_drive_id is not None:
                print(f"Downloading checkpoint from Drive ID {google_drive_id}...")
                try:
                    gdown.download(id=google_drive_id, output=checkpoint_path, quiet=False)
                except Exception as e:
                    print("gdown failed to download weights:", e)

        # Check if the checkpoint exists, and load model_config to override hyperparameters dynamically
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            try:
                # Load with weights_only=False to allow dictionary structure
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    src_vocab_size = config.get('src_vocab_size', src_vocab_size)
                    tgt_vocab_size = config.get('tgt_vocab_size', tgt_vocab_size)
                    d_model = config.get('d_model', d_model)
                    N = config.get('N', N)
                    num_heads = config.get('num_heads', num_heads)
                    d_ff = config.get('d_ff', d_ff)
                    dropout = config.get('dropout', dropout)
            except Exception as e:
                print("Failed to load model_config from checkpoint:", e)

        self.d_model = d_model
        
        # Load vocab files if they exist to determine correct sizes
        self.src_vocab = None
        self.tgt_vocab = None
        if os.path.exists("vocab.pt"):
            try:
                vocab_data = torch.load("vocab.pt", map_location="cpu", weights_only=False)
                self.src_vocab = vocab_data["src_vocab"]
                self.tgt_vocab = vocab_data["tgt_vocab"]
                src_vocab_size = len(self.src_vocab)
                tgt_vocab_size = len(self.tgt_vocab)
            except Exception as e:
                print("Failed to load vocab.pt inside Transformer initialization:", e)
                
        # Load spacy tokenizers inside init
        import spacy
        try:
            self.de_nlp = spacy.load("de_core_news_sm")
        except Exception:
            self.de_nlp = None

        # Build architecture
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, N)
        
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, N)
        
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # Share weight matrices between target embedding and output linear projection
        self.generator.weight = self.tgt_embed.weight

        # Load checkpoint weights if checkpoint exists
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint weights from {checkpoint_path}...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.load_state_dict(checkpoint)
            except Exception as e:
                print("Failed to load checkpoint weights:", e)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # Embed and multiply by sqrt(d_model)
        x = self.src_embed(src) * math.sqrt(self.d_model)
        # Add Positional Encoding
        x = self.pos_encoder(x)
        # Forward through encoder
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # Embed and multiply by sqrt(d_model)
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        # Add Positional Encoding
        x = self.pos_encoder(x)
        # Forward through decoder
        dec_out = self.decoder(x, memory, src_mask, tgt_mask)
        # Project to target vocabulary logits
        return self.generator(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Ensure vocab is loaded
        if self.src_vocab is None or self.de_nlp is None:
            import spacy
            vocab_data = torch.load("vocab.pt", map_location="cpu", weights_only=False)
            self.src_vocab = vocab_data["src_vocab"]
            self.tgt_vocab = vocab_data["tgt_vocab"]
            self.de_nlp = spacy.load("de_core_news_sm")

        # 1. Tokenize the German sentence
        tokens = [tok.text.lower() for tok in self.de_nlp.tokenizer(src_sentence)]
        
        # 2. Convert to token indices
        src_indices = [self.src_vocab.stoi.get('<sos>')] + \
                      [self.src_vocab.stoi.get(tok, self.src_vocab.stoi['<unk>']) for tok in tokens] + \
                      [self.src_vocab.stoi.get('<eos>')]
                      
        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0) # [1, src_len]
        
        # 3. Create src mask
        src_mask = (src_tensor == self.src_vocab.stoi.get('<pad>', 1)).unsqueeze(1).unsqueeze(2) # [1, 1, 1, src_len]
        
        # 4. Run Encoder
        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)
            
            # 5. Greedy decoding
            sos_idx = self.tgt_vocab.stoi.get('<sos>')
            eos_idx = self.tgt_vocab.stoi.get('<eos>')
            pad_idx = self.tgt_vocab.stoi.get('<pad>')
            
            ys = torch.tensor([[sos_idx]], dtype=torch.long, device=device) # [1, 1]
            max_len = 100
            
            for _ in range(max_len - 1):
                tgt_len = ys.size(1)
                tgt_pad_mask = (ys == pad_idx).unsqueeze(1).unsqueeze(2)
                subsequent_mask = torch.triu(torch.ones((1, tgt_len, tgt_len), device=device), diagonal=1).bool().unsqueeze(1)
                tgt_mask = tgt_pad_mask | subsequent_mask
                
                out = self.decode(memory, src_mask, ys, tgt_mask)
                prob = out[:, -1, :]
                next_word = prob.argmax(dim=-1).item()
                
                ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
                
                if next_word == eos_idx:
                    break
                    
        # 6. Convert indices back to tokens and detokenize
        translated_indices = ys.squeeze(0).tolist()
        words = []
        for idx in translated_indices:
            token = self.tgt_vocab.lookup_token(idx)
            if token not in ['<sos>', '<eos>', '<pad>']:
                words.append(token)
                
        translation = " ".join(words)
        return translation
