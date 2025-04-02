"""
State-Resetting Agreement-Based Inference (SR-ABI) for T-JEPA Decoder Model

This script implements the SR-ABI generation strategy for a pre-trained
Transformer Decoder model (specifically, the T-JEPA model architecture trained
on GSM8K or similar tasks).

SR-ABI aims to improve generation consistency and reduce degenerate outputs by:
1.  **State Resetting:** At each generation step, the model's internal state
    (like KV caches, though not explicitly used here as the full context is re-processed)
    is effectively reset by re-evaluating the entire current sequence.
2.  **Voting:** Multiple candidate tokens (K votes) are sampled for the next position
    based on the re-evaluated state.
3.  **Agreement:** The token receiving the most votes (majority vote) is chosen
    as the next token to append to the sequence.

This script includes:
- Configuration settings for inference (model path, prompt, SR-ABI parameters).
- Model definitions copied from the training script (necessary for loading the checkpoint).
- Helper functions for byte-level encoding and decoding.
- The core `generate_sr_abi` function implementing the inference logic.
- A `load_model_for_inference` function to handle checkpoint loading and hyperparameter setup.
- A main execution block to run the inference process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os
import time
from collections import Counter
from typing import Optional, Tuple, List, Dict, Any

# ==========================================
# Configuration
# ==========================================

# --- Paths and Prompts ---
MODEL_PATH = "t_jepa_mtl_decoder_rope_bs1024_checkpoint.pt" # Path to your trained model checkpoint
# MODEL_PATH = "t_jepa_mtl_decoder_rope_bs1024_checkpoint_best.pt" # Or use the best checkpoint
PROMPT_TEXT = "what is five plus two?" # The user prompt for the model
SYSTEM_PROMPT = """Consider this math problem. Think step by step and provide your reasoning between <think> </think> tags, then give your final answer between <answer> </answer> tags.""" # System prompt prepended during formatting
THINK_TAG_START = "<think>" # Tag indicating where the model should start its reasoning

# --- SR-ABI Parameters ---
NUM_VOTES = 8           # K: Number of candidate samples (votes) per token generation step. Higher K = more robust agreement but slower inference.
MAX_NEW_TOKENS = 2048   # L: Maximum number of tokens (bytes) to generate after the prompt.
TEMPERATURE = 0.7       # Sampling temperature for candidate generation. Lower = more deterministic, higher = more random.
TOP_P = 0.9             # Top-p (nucleus) sampling threshold for candidate generation. Limits sampling to most probable tokens.

# ==========================================
# Model Definitions (Copied from training script for loading compatibility)
# ==========================================
# These classes define the model architecture. They must match the architecture
# used during training to allow successful loading of the checkpoint's state_dict.
# The actual forward passes of Predictor/TargetEncoder are NOT used during
# this specific SR-ABI generation process, which only relies on the
# BackboneDecoder and LMHead.

# --- 1) Hyperparameters Default ---
def get_default_hyperparams() -> Dict[str, Any]:
    """
    Provides default hyperparameters. These are used as a fallback if
    hyperparameters are not found within the loaded checkpoint.

    Returns:
        Dict[str, Any]: A dictionary of default hyperparameter values.
    """
    return {
        # Core model structure params
        'vocab_size': 256, 'embed_dim': 512, 'n_heads': 8, 'n_layers': 12,
        'block_size': 1024,
        # Training-related params (needed for model class init, less critical for inference)
        'ema_decay': 0.999, 'lm_loss_weight': 0.1,
        # Special tokens
        'bos_token': 254, 'eos_token': 255, 'pad_token': 0,
        # Default generation params (can be overridden by script config or loaded hyperparams)
        'top_p': 0.8, 'temperature': 0.8,
        # JEPA params (part of model structure definition, not used in generation logic)
        'context_span_ratio': 0.6, 'target_span_ratio': 0.2,
        'num_target_spans': 8, 'min_span_length': 32,
        # Tags (used in prompt formatting and potentially decoding)
        'thinking_tag': "<think>", 'thinking_end_tag': "</think>",
        'answer_tag': "<answer>", 'answer_end_tag': "</answer>",
        'system_prompt': SYSTEM_PROMPT # Use the global constant
    }

# --- 2) RoPE ---
class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE).
    Needed for the BackboneDecoder's attention layers.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000, device: Optional[str] = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves precomputed RoPE frequencies for a given sequence length."""
        # Clamp seq_len instead of raising error if it exceeds max_seq_len during generation
        if seq_len > self.max_seq_len:
             print(f"Warning: RoPE requested seq_len {seq_len} > max_seq_len {self.max_seq_len}. Clamping to {self.max_seq_len}.")
             seq_len = self.max_seq_len
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE rotations to query and key tensors."""
    # Add broadcastable dimensions for batch and head
    cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, T, D_head]
    sin = sin.unsqueeze(0).unsqueeze(0) # [1, 1, T, D_head]
    # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- 3) Attention ---
class ImprovedAttention(nn.Module):
    """
    Multi-Head Attention layer with optional RoPE and causal masking.
    Used within the DecoderBlocks.
    """
    def __init__(self, embed_dim: int, n_heads: int, is_self_attention: bool = True, use_rope: bool = True, max_seq_len: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by n_heads"
        self.is_self_attention = is_self_attention
        self.use_rope = use_rope

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Instantiate RoPE if needed (only for self-attention)
        if self.use_rope and self.is_self_attention:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.rotary_emb = None

        self.attn_dropout = nn.Dropout(0.1) # Dropout on attention weights
        self.out_dropout = nn.Dropout(0.1)  # Dropout on the final output projection
        self.register_buffer("causal_mask_cache", None, persistent=False) # Cache for causal mask

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Gets or creates the causal mask for sequence length T."""
        if self.causal_mask_cache is None or self.causal_mask_cache.shape[-1] < T:
            # Create upper triangular mask (True means mask)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
            self.causal_mask_cache = mask
        return self.causal_mask_cache[:T, :T].to(device=device)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_value_states: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        """Forward pass for the attention layer."""
        B, T, C = x.size() # Batch size, Query sequence length, Embedding dimension
        is_cross_attn = key_value_states is not None
        use_rope_for_this_pass = self.use_rope and self.is_self_attention and not is_cross_attn and self.rotary_emb is not None

        # --- Project Q, K, V ---
        q = self.q_proj(x)
        if is_cross_attn:
            T_k = key_value_states.size(1) # Key/Value sequence length
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
            is_causal = False # Causal mask not used in cross-attention
        else: # Self-attention
            T_k = T
            k = self.k_proj(x)
            v = self.v_proj(x)

        # --- Reshape for Multi-Head ---
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)    # [B, H, T, D_head]
        k = k.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T_k, D_head]
        v = v.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T_k, D_head]

        # --- Apply RoPE ---
        if use_rope_for_this_pass:
            cos, sin = self.rotary_emb(T) # Get embeddings for query length T
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            scaling_factor = 1.0 # RoPE often doesn't need 1/sqrt(dk) scaling
        else:
            scaling_factor = 1.0 / math.sqrt(self.head_dim) # Standard scaling

        # --- Compute Scores ---
        # scores: [B, H, T, T_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling_factor

        # --- Apply Masking (Padding and Causal) ---
        final_mask_bool = None
        # Padding mask
        if attn_mask is not None:
            # Ensure mask is boolean where True means "mask this position"
            if attn_mask.dim() == 2: padding_mask_bool = ~attn_mask.bool().unsqueeze(1).unsqueeze(2) # [B, T_k] -> [B, 1, 1, T_k]
            elif attn_mask.dim() == 4: padding_mask_bool = ~attn_mask.bool() # Assume already broadcastable
            else: raise ValueError(f"Unsupported attn_mask dimension: {attn_mask.dim()}")
            final_mask_bool = padding_mask_bool
        # Causal mask (only for self-attention)
        if self.is_self_attention and is_causal:
            causal_mask_bool = self._get_causal_mask(T, x.device).unsqueeze(0).unsqueeze(0) # [T, T] -> [1, 1, T, T]
            if final_mask_bool is not None: final_mask_bool = final_mask_bool | causal_mask_bool # Combine masks
            else: final_mask_bool = causal_mask_bool
        # Apply the combined mask
        if final_mask_bool is not None:
             scores = scores.masked_fill(final_mask_bool, torch.finfo(scores.dtype).min)

        # --- Softmax, Dropout, Output ---
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v) # [B, H, T, D_head]
        # Reshape back to [B, T, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_dropout(self.out_proj(attn_output))

# --- 4) Decoder Block ---
class DecoderBlock(nn.Module):
    """
    A single Transformer Decoder block with Pre-LN structure.
    Contains self-attention and a feed-forward network.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=True, use_rope=True, max_seq_len=max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = 4 * embed_dim # Standard expansion factor
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout) # Dropout for residual connections

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, is_causal: bool = True) -> torch.Tensor:
        """Forward pass for the decoder block."""
        # Self-Attention part (LN -> Attention -> Dropout -> Residual)
        residual = x
        x_norm = self.ln1(x)
        attn_output = self.self_attention(x_norm, attn_mask=attention_mask, is_causal=is_causal)
        x = residual + self.dropout(attn_output)
        # Feed-Forward part (LN -> FFN -> Dropout -> Residual)
        residual = x
        x_norm = self.ln2(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)
        return x

# --- 5) JEPA Predictor Block (Needed for model structure, not used in generation logic) ---
class JEPAPredictorBlock(nn.Module):
    """
    A block for the JEPA Predictor. Includes self-attention and cross-attention.
    Required for loading the model state_dict, but its forward logic is not
    executed during SR-ABI generation.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=True, use_rope=True, max_seq_len=max_seq_len)
        self.ln_cross_attn_query = nn.LayerNorm(embed_dim)
        self.ln_cross_attn_kv = nn.LayerNorm(embed_dim)
        self.cross_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=False, use_rope=False, max_seq_len=max_seq_len)
        self.ln3 = nn.LayerNorm(embed_dim)
        hidden_dim = 4 * embed_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, decoder_output: torch.Tensor, self_attention_mask: Optional[torch.Tensor] = None, cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass (not used in SR-ABI generation)."""
        # This function's logic is complex and specific to JEPA training.
        # Since it's not needed for SR-ABI inference, we can skip implementing its body
        # as long as the layers are defined in __init__ for state_dict loading.
        raise NotImplementedError("JEPAPredictorBlock forward pass is not needed for SR-ABI generation.")

# --- 6) Backbone Decoder ---
class BackboneDecoder(nn.Module):
    """
    The main Transformer Decoder stack used for processing sequences.
    This is the core component used during generation.
    """
    def __init__(self, vocab_size: int, embed_dim: int, n_heads: int, n_layers: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        # Input embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.1) # Dropout after embedding
        # Stack of decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, dropout=0.1, max_seq_len=block_size)
            for _ in range(n_layers)
        ])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(embed_dim)
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights for Linear and Embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
             # Initialize LayerNorm gain=1, bias=0
             torch.nn.init.ones_(module.weight)
             torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, is_causal: bool = True) -> torch.Tensor:
        """
        Forward pass through the Backbone Decoder.

        Args:
            x (torch.Tensor): Input token indices [B, T].
            attention_mask (Optional[torch.Tensor]): Padding mask [B, T]. 1=keep, 0=mask.
            is_causal (bool): Whether to use causal masking in self-attention. True for generation.

        Returns:
            torch.Tensor: Output embeddings [B, T, C].
        """
        B, T = x.size()
        # Note: We don't assert T <= self.block_size here, as the calling function
        # handles cropping the input sequence `x` before passing it.

        # Get token embeddings + dropout
        token_emb = self.token_embedding(x) # [B, T, C]
        h = self.dropout(token_emb)

        # Pass through all decoder blocks
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask, is_causal=is_causal)

        # Final layer normalization
        h = self.ln_f(h) # [B, T, C]
        return h

# --- 7) JEPA Predictor (Needed for model structure, not used in generation logic) ---
class JEPAPredictor(nn.Module):
    """
    JEPA Predictor module structure.
    Required for loading the model state_dict, but its forward logic is not
    executed during SR-ABI generation.
    """
    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        predictor_layers = n_layers # Use same number of layers as backbone by default
        # Learnable mask token embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        # Predictor blocks
        self.blocks = nn.ModuleList([
            JEPAPredictorBlock(embed_dim, n_heads, max_seq_len=block_size)
            for _ in range(predictor_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim) # Final layer norm
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights for Linear and LayerNorm layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
             torch.nn.init.ones_(module.weight)
             torch.nn.init.zeros_(module.bias)

    def forward(self, decoder_output_causal: torch.Tensor, target_spans_indices: List, context_mask: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass (not used in SR-ABI generation)."""
        raise NotImplementedError("JEPAPredictor forward pass is not needed for SR-ABI generation.")

# --- 8) Target Encoder (Needed for model structure, not used in generation logic) ---
class TargetEncoder(nn.Module):
    """
    Target Encoder module structure (EMA copy of Backbone).
    Required for loading the model state_dict, but its forward logic and EMA updates
    are not executed during SR-ABI generation.
    """
    def __init__(self, backbone_decoder: BackboneDecoder, ema_decay: float = 0.999):
        super().__init__()
        # Deep copy structure and initial weights
        self.encoder = copy.deepcopy(backbone_decoder)
        self.ema_decay = ema_decay
        # Disable gradients for target encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_ema(self, backbone_decoder: BackboneDecoder, decay_rate: Optional[float] = None):
        """EMA update logic (not used in SR-ABI generation)."""
        pass # No EMA updates needed during inference

    @torch.no_grad()
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass (not used in SR-ABI generation)."""
        raise NotImplementedError("TargetEncoder forward pass is not needed for SR-ABI generation.")

# --- 9) Complete T-JEPA Model ---
class TJEPAModel(nn.Module):
    """
    The complete T-JEPA Model class.

    This class integrates the BackboneDecoder, JEPAPredictor, TargetEncoder,
    and LMHead. For SR-ABI inference, only the `decoder_backbone` and `lm_head`
    are actively used in the generation loop. The other components (Predictor,
    TargetEncoder) are included here primarily to ensure that the model's
    state_dict can be loaded correctly from a checkpoint saved during training.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of embeddings and hidden states.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of layers in the backbone and predictor.
        block_size (int): Maximum sequence length (context window).
        ema_decay (float): EMA decay for TargetEncoder (structural only for inference).
        lm_loss_weight (float): LM loss weight (structural only for inference).
        pad_token_id (int): ID of the padding token.
    """
    def __init__(self, vocab_size: int, embed_dim: int, n_heads: int, n_layers: int, block_size: int, ema_decay: float = 0.999, lm_loss_weight: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.lm_loss_weight = lm_loss_weight # Stored, but not used in loss calc here
        self.block_size = block_size

        # --- Core Components for Generation ---
        self.decoder_backbone = BackboneDecoder(vocab_size, embed_dim, n_heads, n_layers, block_size)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # --- Components for Structural Compatibility ---
        # These are needed so that `load_state_dict` finds matching keys from the training checkpoint.
        self.predictor = JEPAPredictor(embed_dim, n_heads, n_layers, block_size)
        self.target_encoder = TargetEncoder(self.decoder_backbone, ema_decay)

        # --- Weight Tying ---
        # Tie the embedding weights and the final LM head weights
        self.decoder_backbone.token_embedding.weight = self.lm_head.weight
        print("Weight tying applied between token embedding and LM head.")

    # Note: The main `forward`, `compute_loss`, `update_target_encoder`, and standard `generate`
    # methods from the training script are omitted here, as SR-ABI uses a custom generation loop
    # that directly calls the `decoder_backbone` and `lm_head`.

# ==========================================
# Helper Functions for Tokenization
# ==========================================

def _encode(text: str, bos_token: int) -> List[int]:
    """
    Encodes a string into a list of byte tokens (integers), prepending the BOS token.

    Args:
        text (str): The input string.
        bos_token (int): The integer ID for the Beginning-of-Sequence token.

    Returns:
        List[int]: A list of byte token IDs, including the initial BOS token.
    """
    return [bos_token] + list(text.encode('utf-8', errors='replace'))

def _decode(tokens: List[int], bos_token: int, eos_token: int, pad_token: int) -> str:
    """
    Decodes a list of byte tokens (integers) back into a string.
    Handles truncation at the first EOS token and removes BOS/PAD tokens.

    Args:
        tokens (List[int]): The list of token IDs to decode.
        bos_token (int): The integer ID for the BOS token.
        eos_token (int): The integer ID for the End-of-Sequence token.
        pad_token (int): The integer ID for the Padding token.

    Returns:
        str: The decoded string. Returns an error message if decoding fails.
    """
    try:
        # Find the first occurrence of EOS token, if present
        eos_pos = -1
        if eos_token in tokens:
            eos_pos = tokens.index(eos_token)

        # Truncate the list at the EOS token
        if eos_pos != -1:
            tokens = tokens[:eos_pos]

        # Filter out special tokens (BOS, PAD) and convert remaining ints to bytes
        filtered_bytes = bytes([tok for tok in tokens if tok != bos_token and tok != pad_token])

        # Decode the byte sequence using UTF-8, replacing errors
        return filtered_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Error during decoding: {e}")
        # Provide raw byte representation as fallback
        return f"[Decoding Error] Raw bytes: {bytes(tokens)}"

# ==========================================
# SR-ABI Inference Function
# ==========================================

@torch.no_grad() # Disable gradient calculations for inference
def generate_sr_abi(
    model: TJEPAModel,
    prompt_text: str,
    num_votes: int,           # K parameter for SR-ABI
    max_new_tokens: int,      # L parameter (max generation length)
    temperature: float,       # Sampling temperature (Theta)
    top_p: float,             # Nucleus sampling threshold (Theta)
    hyperparams: Dict[str, Any], # Dictionary containing special tokens, block_size, etc.
    device: str               # Compute device ('cuda', 'cpu', 'mps')
) -> str:
    """
    Generates text using State-Resetting Agreement-Based Inference (SR-ABI).

    At each step, it samples K candidate next tokens by re-evaluating the current
    sequence, takes a majority vote among the candidates, and appends the winner.

    Args:
        model (TJEPAModel): The loaded T-JEPA model instance.
        prompt_text (str): The initial user prompt.
        num_votes (int): K, the number of candidate samples per step.
        max_new_tokens (int): L, the maximum number of tokens to generate.
        temperature (float): Sampling temperature for candidate generation.
        top_p (float): Top-p (nucleus) sampling threshold.
        hyperparams (Dict[str, Any]): Loaded hyperparameters including special token IDs
                                      (bos, eos, pad), block_size, system_prompt.
        device (str): The device to run inference on.

    Returns:
        str: The generated text, including the formatted prompt.
    """
    model.eval() # Ensure model is in evaluation mode

    # --- Get necessary parameters from hyperparams ---
    bos_token = hyperparams['bos_token']
    eos_token = hyperparams['eos_token']
    pad_token = hyperparams['pad_token']
    block_size = hyperparams['block_size']
    system_prompt = hyperparams.get('system_prompt', "") # Get system prompt, default to empty if missing

    # --- Initialization ---
    # a. Format the full prompt including system prompt and starting tag
    full_prompt = f"{system_prompt}\n\nProblem: {prompt_text}\n\n{THINK_TAG_START}"
    # Tokenize the formatted prompt
    prompt_tokens = _encode(full_prompt, bos_token)

    # b. Initialize the current full sequence S (list of token IDs)
    S_list = prompt_tokens[:] # Start with the prompt tokens

    # c. Initialize the generated sequence G (list of token IDs) - only the new part
    G_list = []

    print(f"\n--- Starting SR-ABI Generation (K={num_votes}, Max Tokens={max_new_tokens}) ---")
    print(f"Formatted Prompt:\n{full_prompt}", end="", flush=True) # Print the prompt without trailing newline

    # --- Token Generation Loop ---
    # Loop for a maximum of `max_new_tokens` steps
    for i in range(max_new_tokens):
        # --- a. Vote Collection ---
        votes = Counter() # Use Counter to store votes {token_id: count}
        # Prepare the current sequence S as a tensor for the model input
        S_tensor = torch.tensor([S_list], dtype=torch.long, device=device) # Add batch dimension: [1, current_len]

        # Crop the sequence if it exceeds the model's block size
        # Take the last `block_size` tokens as context for the forward pass
        S_cond = S_tensor if S_tensor.size(1) <= block_size else S_tensor[:, -block_size:]
        current_context_len = S_cond.size(1)

        # Create the attention mask for the (potentially cropped) context
        # Mask value is 1.0 for real tokens, 0.0 for padding (though padding shouldn't be in S_list)
        attention_mask = (S_cond != pad_token).float().to(device) # Shape: [1, current_context_len]

        # Sample K candidates for the next token
        for j in range(num_votes):
            # --- i.1 & i.2: Reset State & Re-evaluate Context ---
            # By passing the *entire current context* (S_cond) to the backbone in each
            # iteration of the voting loop, we effectively reset the model's state
            # (equivalent to clearing KV cache if it were used explicitly) and force
            # re-computation based on the full history up to this point.
            decoder_output = model.decoder_backbone(
                S_cond,
                attention_mask=attention_mask,
                is_causal=True # Causal masking is essential for autoregressive generation
            ) # Output shape: [1, current_context_len, embed_dim]

            # Get logits for the *next* token by taking the output embedding of the *last* token
            # in the current context sequence.
            # decoder_output[:, -1, :] gives shape [1, embed_dim]
            logits = model.lm_head(decoder_output[:, -1, :])  # Project to vocab size: [1, vocab_size]

            # --- i.3: Sample Candidate Token ---
            # Apply temperature scaling to logits
            if temperature > 0 and temperature != 1.0:
                 logits = logits / temperature
            # Apply top-p (nucleus) sampling to filter logits
            if top_p > 0.0 and top_p < 1.0:
                # Sort logits, calculate cumulative probability
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Create mask for tokens outside the nucleus
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # Keep the first token over threshold
                sorted_indices_to_remove[..., 0] = 0 # Always keep the most probable token
                # Scatter mask back to original indices
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                # Apply mask by setting low-probability logits to -infinity
                logits[indices_to_remove] = float('-inf')

            # Convert final logits to probabilities
            probs = F.softmax(logits, dim=-1) # Shape: [1, vocab_size]
            # Sample one candidate token based on the probability distribution
            candidate_token_tensor = torch.multinomial(probs, num_samples=1) # Shape: [1, 1]
            candidate_token = candidate_token_tensor.item() # Get the integer token ID

            # --- i.4: Record Vote ---
            votes[candidate_token] += 1

        # --- b. Agreement (Select Winning Token) ---
        # Check if any votes were collected (should always be true if num_votes >= 1)
        if not votes:
            print("\nWarning: No votes were collected in this step. Stopping generation.", flush=True)
            break

        # Find the token ID with the highest vote count (majority vote)
        # `most_common(1)` returns a list like [(token_id, count)]
        winning_token, vote_count = votes.most_common(1)[0]

        # Optional: Log the voting process for debugging
        # print(f"\n[Step {i+1}] Votes: {dict(votes)}")
        # print(f" -> Winner: {winning_token} (Byte: {bytes([winning_token]) if 0<=winning_token<=255 else 'Special'}) with {vote_count}/{num_votes} votes")

        # --- c. Check for Termination Condition ---
        # If the winning token is the End-of-Sequence token, stop generation.
        if winning_token == eos_token:
            print(" <EOS>", flush=True) # Indicate EOS was reached
            break

        # --- d. Append Winning Token ---
        # Add the winning token to the list of *generated* tokens (G)
        G_list.append(winning_token)
        # Add the winning token to the *full* sequence list (S) for the next step's context
        S_list.append(winning_token)

        # --- Print the winning token (for streaming effect) ---
        # Attempt to decode the single winning byte token immediately.
        # Handle potential UnicodeDecodeError if it's part of a multi-byte sequence.
        try:
            # Wrap the single byte token ID in a list and convert to bytes object
            byte_to_decode = bytes([winning_token])
            print(byte_to_decode.decode('utf-8', errors='replace'), end="", flush=True)
        except UnicodeDecodeError:
             # If it's part of a multi-byte char, print a placeholder or nothing,
             # the full character will be printed once all bytes are received.
             print("<?>", end="", flush=True) # Placeholder for undecodable byte fragment
        except Exception as print_e:
             print(f"[Print Error: {print_e}]", end="", flush=True) # Catch other potential errors

        # Optional: Small delay for better visualization of token streaming
        # time.sleep(0.01)

    print("\n--- Generation Complete ---") # Add a newline after the generation loop finishes

    # --- Finalization ---
    # a. Detokenize the generated sequence G (excluding the prompt)
    # Use the helper function to decode the list of generated token IDs (G_list)
    output_text_generated_part = _decode(G_list, bos_token, eos_token, pad_token)

    # Combine the original formatted prompt with the decoded generated text
    final_output_text = full_prompt + output_text_generated_part

    return final_output_text

# ==========================================
# Model Loading Function
# ==========================================
def load_model_for_inference(model_path: str, device: str) -> Tuple[Optional[TJEPAModel], Optional[Dict[str, Any]]]:
    """
    Loads the T-JEPA model and its hyperparameters from a checkpoint file.

    Handles potential issues like missing hyperparameters in the checkpoint
    by falling back to defaults, and uses flexible state_dict loading.

    Args:
        model_path (str): Path to the model checkpoint (.pt) file.
        device (str): The compute device ('cuda', 'cpu', 'mps').

    Returns:
        Tuple[Optional[TJEPAModel], Optional[Dict[str, Any]]]: A tuple containing:
            - The loaded model instance (or None if loading fails).
            - The effective hyperparameters dictionary used (or None if loading fails).
    """
    # Check if the checkpoint file exists
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint file not found at '{model_path}'")
        return None, None

    print(f"Loading model checkpoint from '{model_path}'...")
    try:
        # Load the checkpoint data onto the specified device
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint file '{model_path}': {e}")
        return None, None

    # --- Determine Hyperparameters ---
    # Try loading hyperparameters saved within the checkpoint
    hyperparams_loaded = checkpoint.get('hyperparams', None)
    if hyperparams_loaded:
        print("  Hyperparameters found in checkpoint.")
        # Start with defaults to ensure all keys exist, then update with loaded values
        hyperparams = get_default_hyperparams()
        hyperparams.update(hyperparams_loaded) # Loaded values overwrite defaults if present
    else:
        # If no hyperparameters found in checkpoint, use the defaults
        print("  Warning: Hyperparameters not found in checkpoint. Using default values.")
        hyperparams = get_default_hyperparams()

    # Log the final hyperparameters being used (useful for debugging)
    print("  Effective Hyperparameters for Inference:")
    # Print neatly for readability
    for key, val in hyperparams.items():
          print(f"    {key}: {val}")

    # --- Create Model Instance ---
    print("  Initializing model structure...")
    try:
        # Instantiate the TJEPAModel using the determined hyperparameters
        model = TJEPAModel(
            vocab_size=hyperparams['vocab_size'],
            embed_dim=hyperparams['embed_dim'],
            n_heads=hyperparams['n_heads'],
            n_layers=hyperparams['n_layers'],
            block_size=hyperparams['block_size'],
            # Pass other required params from hyperparams dict
            ema_decay=hyperparams.get('ema_decay', 0.999), # Provide default if missing
            lm_loss_weight=hyperparams.get('lm_loss_weight', 0.1), # Provide default if missing
            pad_token_id=hyperparams['pad_token']
        ).to(device)
    except KeyError as e:
         # Error if essential hyperparameter is missing from both checkpoint and defaults
         print(f"  Error: Missing essential hyperparameter '{e}' needed to build the model structure.")
         return None, None
    except Exception as e:
        print(f"  Error creating model instance: {e}")
        return None, None

    # --- Load Model State Weights ---
    print("  Loading model state dictionary...")
    try:
        model_state = checkpoint['model_state']

        # Use flexible loading (non-strict) to handle potential minor architecture changes
        # or differences between training/inference model definitions (e.g., missing Predictor weights)
        current_model_dict = model.state_dict()
        processed_state_dict = {} # State dict to actually load
        warned_keys_shape = set() # Track keys with shape mismatches
        warned_keys_missing_in_model = set() # Track keys in checkpoint but not model
        loaded_keys_count = 0

        for k, v in model_state.items():
            new_k = k # Placeholder for potential key renaming logic if needed in future
            # Example rename: if k.startswith("old_prefix."): new_k = k.replace("old_prefix.", "new_prefix.", 1)

            if new_k in current_model_dict:
                # Check if shapes match before adding to the dict to load
                if v.shape == current_model_dict[new_k].shape:
                    processed_state_dict[new_k] = v
                    loaded_keys_count += 1
                else:
                    # Log shape mismatch warning only once per key
                    if new_k not in warned_keys_shape:
                        print(f"    Warning: Shape mismatch for key '{new_k}'. Checkpoint: {v.shape}, Model: {current_model_dict[new_k].shape}. Skipping this weight.")
                        warned_keys_shape.add(new_k)
            else:
                # Log warning if a key from the checkpoint isn't in the current model structure
                original_key_str = f" (original: '{k}')" if new_k != k else ""
                if new_k not in warned_keys_missing_in_model:
                     # print(f"    Info: Key '{new_k}'{original_key_str} from checkpoint not found in current model. Skipping.")
                     warned_keys_missing_in_model.add(new_k)
                     if k != new_k: warned_keys_missing_in_model.add(k)

        # Load the processed state dict using strict=False
        missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)

        # Report keys that were in the model but not found in the processed checkpoint state
        if missing_keys:
            # This is expected for Predictor/TargetEncoder if they weren't in the checkpoint's processed_state_dict
            print(f"    Info: Weights for some keys were missing in the checkpoint and remain initialized: {missing_keys}")
        # Report keys loaded from checkpoint but not expected by model (should be empty due to filtering above)
        if unexpected_keys:
             print(f"    Warning: Some weights from the checkpoint were unexpected by the model: {unexpected_keys}")

        print(f"  Model state loaded successfully ({loaded_keys_count} tensors assigned).")

        # Print details from the checkpoint for context
        loaded_epoch = checkpoint.get('epoch', 'N/A')
        loaded_step = checkpoint.get('current_step', 'N/A')
        val_loss = checkpoint.get('val_loss', 'N/A')
        # Format validation loss nicely if it's a float
        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
        print(f"    Checkpoint Source: Epoch={loaded_epoch}, Step={loaded_step}, Val Loss={val_loss_str}")

    except KeyError:
        # Handle case where 'model_state' key is missing in the checkpoint
        print("  Error: 'model_state' key not found in the checkpoint file.")
        print("  Attempting inference with initialized model weights (results may be poor).")
    except Exception as e:
        print(f"  Error loading model state weights: {e}")
        print("  Attempting inference with potentially uninitialized weights.")

    # Return the loaded model and the effective hyperparameters
    return model, hyperparams

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    print("============================================")
    print(" SR-ABI Inference Script for T-JEPA Model ")
    print("============================================")

    # --- Setup Device ---
    # Automatically select MPS (Apple Silicon GPU), CUDA (NVIDIA GPU), or CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using compute device: {device}")

    # --- Load Model and Hyperparameters ---
    # Call the loading function, which handles checkpoint reading, hyperparam setup,
    # model instantiation, and state dict loading.
    model, hyperparams = load_model_for_inference(MODEL_PATH, device)

    # Proceed only if model loading was successful
    if model and hyperparams:
        print("\nModel loaded successfully. Starting SR-ABI generation...")

        # --- Run SR-ABI Generation ---
        start_time = time.time() # Start timer
        generated_text = generate_sr_abi(
            model=model,
            prompt_text=PROMPT_TEXT,        # User prompt defined in config section
            num_votes=NUM_VOTES,            # K parameter defined in config
            max_new_tokens=MAX_NEW_TOKENS,  # L parameter defined in config
            temperature=TEMPERATURE,        # Sampling temp defined in config
            top_p=TOP_P,                    # Top-p sampling defined in config
            hyperparams=hyperparams,        # Pass loaded/effective hyperparams
            device=device                   # Pass selected compute device
        )
        end_time = time.time() # End timer

        # --- Print Final Output and Timing ---
        # The `generate_sr_abi` function prints token-by-token during generation.
        # Here, we print the complete final string again for clarity.
        print("\n\n--- Final Generated Output ---")
        print(generated_text)
        print("------------------------------")
        print(f"SR-ABI generation took {end_time - start_time:.2f} seconds.")
    else:
        # If model loading failed
        print("\nModel loading failed. Cannot proceed with inference. Exiting.")
        exit(1) # Exit with error code

    print("\nScript execution finished.")
