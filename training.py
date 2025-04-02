import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from typing import Optional, Tuple

# ==========================================
# 1) Hyperparameters
# ==========================================
def get_hyperparams():
    return {
        # Model Parameters
        'batch_size': 2,
        'block_size': 1024,               # INCREASED - More space for spans
        'vocab_size': 256,
        'embed_dim': 512,
        'n_heads': 8,
        'n_layers': 12,                    # Number of Decoder Blocks

        # JEPA Parameters
        'context_span_ratio': 0.6,        # Ratio calculation might need tuning with larger block size
        'target_span_ratio': 0.2,         # Ratio calculation might need tuning with larger block size
        'num_target_spans': 8,            # DECREASED - More realistic number
        'min_span_length': 32,            # DECREASED - Easier to fit smaller spans

        # Training Parameters
        'num_epochs': 50,
        'steps_per_epoch': 1000,
        'eval_interval': 200,
        'eval_iters': 100,
        'ema_decay': 0.999,
        'accumulation_steps': 8,
        'lm_loss_weight': 0.92,

        # Special Tokens
        'bos_token': 254,
        'eos_token': 255,
        'pad_token': 0,

        # Generation Parameters
        'generate_num_tokens': 1024,      # Can match block_size or be different
        'top_p': 0.8,
        'start_prompt': "Problem: A bakery produces cakes for $10 each. It costs them $5 in ingredients per cake, and they have a fixed overhead of $200 per day. How many cakes do they need to sell each day to make a daily profit of $100?",

        # Special Tags
        'thinking_tag': "<think>",
        'thinking_end_tag': "</think>",
        'answer_tag': "<answer>",
        'answer_end_tag': "</answer>",

        # Paths & Modes
        'checkpoint_path': "t_jepa_mtl_decoder_rope_bs1024_checkpoint.pt", # Updated name for new block size
        'continue_training': True,
        'system_prompt': """Consider this math problem. Think step by step and provide your reasoning between <think> </think> tags, then give your final answer between <answer> </answer> tags."""
    }

# ==========================================
# 1.1) Select device
# ==========================================
def get_device():
    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# ==========================================
# 1.2) Data Loading and Preprocessing for GSM8K
# ==========================================
def load_gsm8k_data():
    print("Loading GSM8K dataset...")

    try:
        # Try using the 'datasets' library first
        from datasets import load_dataset
        # Specify cache_dir to avoid potential permission issues in default locations
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        os.makedirs(cache_dir, exist_ok=True)
        dataset = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
        train_df = dataset["train"].to_pandas()
        test_df = dataset["test"].to_pandas()
        print("Dataset loaded using datasets library")
    except Exception as e:
        print(f"Error loading dataset with datasets library: {e}")
        print("Attempting alternative loading methods...")
        try:
            # Alternative: Load directly from Hugging Face Hub parquet files
            print("Attempting to load from Hugging Face Hub parquet files...")
            # Ensure you have pyarrow and fsspec installed: pip install pyarrow fsspec aiohttp
            splits = {'train': 'main/train-00000-of-00001.parquet',
                      'test': 'main/test-00000-of-00001.parquet'}
            train_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
            test_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])
            print("Dataset loaded using parquet files from Hugging Face Hub")
        except Exception as e2:
            print(f"Failed to load dataset using parquet from Hub: {e2}")
            # Fallback to local path if available (adjust path if needed)
            local_path_train = "./gsm8k_data/train.jsonl" # Example local path
            local_path_test = "./gsm8k_data/test.jsonl"   # Example local path
            if os.path.exists(local_path_train) and os.path.exists(local_path_test):
                 print("Attempting to load from local JSONL files...")
                 train_df = pd.read_json(local_path_train, lines=True)
                 test_df = pd.read_json(local_path_test, lines=True)
                 print("Dataset loaded from local JSONL files.")
            else:
                print(f"Local files not found at {local_path_train} and {local_path_test}")
                raise RuntimeError("Unable to load the GSM8K dataset via datasets, parquet, or local files.")


    print(f"Training examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")

    # Split training data into train/validation
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    print(f"Final training examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print(f"Test examples: {len(test_df)}")

    return train_df, val_df, test_df


# ==========================================
# 1.3) Prepare data for JEPA training
# ==========================================
def prepare_batches_from_gsm8k(data_df, hyperparams, device):
    """Create training batches from GSM8K dataset with context and target spans for JEPA."""
    batch_indices = torch.randint(0, len(data_df), (hyperparams['batch_size'],))
    batch_examples = data_df.iloc[batch_indices]

    block_size = hyperparams['block_size']
    bos_token = hyperparams['bos_token']
    eos_token = hyperparams['eos_token']
    pad_token = hyperparams['pad_token']

    # JEPA specific parameters
    num_target_spans = hyperparams['num_target_spans']
    min_span_length = hyperparams['min_span_length']

    # Create storage for batches
    full_sequences = []
    context_masks = [] # Mask for JEPA context (1=context, 0=target/padding)
    target_spans_indices = [] # List of (start, end) tuples

    for _, row in batch_examples.iterrows():
        # Get question and answer
        question = row['question']
        answer = row['answer']

        # Format with system prompt and tags
        system_prompt = hyperparams['system_prompt']
        full_text = f"{system_prompt}\n\nProblem: {question}\n\n<think>{answer}</think>\n\n<answer>"

        # Extract the final answer from the explanation
        answer_lines = answer.strip().split('\n')
        final_answer = answer_lines[-1] if answer_lines else ""
        if "answer is" in final_answer.lower():
            final_answer = final_answer.split("answer is")[-1].strip()
        elif "=" in final_answer:
            final_answer = final_answer.split("=")[-1].strip()
        # Simple extraction, might need refinement
        final_answer_numeric = ''.join(filter(lambda x: x.isdigit() or x == '.', final_answer.split('####')[-1].strip()))

        full_text += f"{final_answer_numeric}</answer>"

        # Convert to byte sequence and add BOS/EOS tokens
        full_bytes = [bos_token] + [b for b in full_text.encode('utf-8', errors='replace')] + [eos_token]

        # Truncate or pad sequence to block_size
        seq_length = len(full_bytes)
        if seq_length > block_size:
            full_bytes = full_bytes[:block_size]
            seq_length = block_size # Actual length after potential truncation
        elif seq_length < block_size:
            padding_needed = block_size - seq_length
            full_bytes = full_bytes + [pad_token] * padding_needed
            # seq_length remains the original length before padding

        # Create context mask (1 = keep as context, 0 = mask for JEPA prediction)
        # Initialize all non-padding positions as potential context
        context_mask = torch.zeros(block_size, dtype=torch.float) # Use float for easier masking later
        context_mask[:seq_length] = 1 # Only real tokens can be context initially

        # Select random target spans for this example
        current_target_spans = []
        # Indices of real tokens available for masking
        available_indices = torch.where(context_mask[:seq_length] == 1)[0].tolist()

        for _ in range(num_target_spans):
            # Check if enough *remaining* tokens are available to form a min_span_length span
            if len(available_indices) < min_span_length:
                break # Not enough remaining tokens

            # Randomly choose target span length
            # Max length: limited by available indices and a fraction of total real tokens
            max_possible_len = min(len(available_indices), int(seq_length * 0.2)) # e.g., Max 20% of real tokens
            if max_possible_len < min_span_length:
                 continue # Skip if max possible length is too small

            # Ensure span_length > min_span_length
            span_length = torch.randint(min_span_length, max(min_span_length + 1, max_possible_len + 1), (1,)).item()

            # Choose random starting position *from the list of available indices*
            if len(available_indices) - span_length < 0:
                # This shouldn't happen if max_possible_len logic is correct, but safety check
                continue
            start_idx_in_available = torch.randint(0, len(available_indices) - span_length + 1, (1,)).item()
            start_pos = available_indices[start_idx_in_available]

            # Calculate end position based on start_pos and span_length
            # Ensure span doesn't exceed sequence length (should be covered by available_indices logic)
            end_pos = min(start_pos + span_length, seq_length)
            actual_span_length = end_pos - start_pos

            # Skip very short spans that might result from hitting the seq_length boundary
            if actual_span_length < min_span_length // 2:
                continue

            # Mark positions in target span on the context mask (set to 0)
            context_mask[start_pos:end_pos] = 0

            # Store span positions (start, end)
            current_target_spans.append((start_pos, end_pos))

            # Update available indices: remove indices used by the target span
            new_available_indices = []
            span_indices_set = set(range(start_pos, end_pos))
            for idx in available_indices:
                if idx not in span_indices_set:
                    new_available_indices.append(idx)
            available_indices = new_available_indices
            # Check if we successfully removed indices
            # print(f" Span {start_pos}-{end_pos}, Remaining indices: {len(available_indices)}") # Debug


        # Add to batches
        full_sequences.append(full_bytes)
        context_masks.append(context_mask)
        target_spans_indices.append(current_target_spans)
        # if not current_target_spans: print(f"Warning: No spans generated for an example. SeqLen: {seq_length}") # Debug

    # Convert to tensors
    x = torch.tensor(full_sequences, dtype=torch.long).to(device)
    context_masks = torch.stack(context_masks).to(device) # [B, T], 1 for context, 0 for target/padding

    # Create attention mask (1 for real tokens including targets, 0 for padding)
    # This mask is used by all attention layers to ignore padding
    attention_mask = (x != pad_token).float().to(device) # [B, T], 1 for non-pad, 0 for pad

    return x, context_masks, target_spans_indices, attention_mask


# ==========================================
# 2) Rotary Positional Embedding (RoPE)
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        # Adjust max_seq_len for RoPE based on the actual block_size
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        # x: [*, seq_len, *]
        # returns: cos, sin buffers of shape [seq_len, dim]
        # Handle cases where seq_len might exceed precomputed length during generation potentially
        if seq_len > self.max_seq_len:
             # Dynamically extend RoPE if needed (more complex, often avoided by setting max_seq_len large enough)
             # For now, we assume seq_len <= self.max_seq_len based on block_size
             # Or simply clamp:
             # print(f"Warning: RoPE seq_len {seq_len} > max_seq_len {self.max_seq_len}. Clamping.")
             # seq_len = self.max_seq_len
            raise ValueError(f"RoPE sequence length {seq_len} exceeds precomputed max {self.max_seq_len}")

        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies RoPE to query and key tensors."""
    # Add sequence length dimension if necessary
    cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, T, D_head]
    sin = sin.unsqueeze(0).unsqueeze(0) # [1, 1, T, D_head]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==========================================
# 3) Improved Attention Mechanism (with RoPE and Causal Masking)
# ==========================================
class ImprovedAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, is_self_attention=True, use_rope=True, max_seq_len=2048):
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

        # Instantiate RoPE only if used and needed
        if self.use_rope and self.is_self_attention:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.rotary_emb = None

        self.attn_dropout = nn.Dropout(0.1)
        self.out_dropout = nn.Dropout(0.1)

        # Buffer for causal mask (recreated if needed)
        self.register_buffer("causal_mask_cache", None, persistent=False)

    def _get_causal_mask(self, T, device):
        # Efficiently get or create causal mask
        if self.causal_mask_cache is None or self.causal_mask_cache.shape[-1] < T:
            # Create lower triangular mask (True for positions to be masked)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
            self.causal_mask_cache = mask
        # Return the sub-mask for the current sequence length T
        # Ensure it's on the correct device (might change between train/eval/inference)
        return self.causal_mask_cache[:T, :T].to(device=device)


    def forward(self, x, attn_mask=None, key_value_states=None, is_causal=False):
        """
        Args:
            x: Query input [B, T, C]
            attn_mask: Padding mask [B, T_k] or broadcastable. 1=Keep, 0=Mask.
            key_value_states: Optional key/value input for cross-attention [B, T_k, C].
            is_causal: If True, apply causal masking (for self-attention only).
        """
        B, T, C = x.size()
        is_cross_attn = key_value_states is not None
        # Determine if RoPE should be applied in this specific call
        use_rope_for_this_pass = self.use_rope and self.is_self_attention and not is_cross_attn and self.rotary_emb is not None

        # Project query
        q = self.q_proj(x)

        # Project keys and values
        if is_cross_attn:
            T_k = key_value_states.size(1)
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
            # Causal mask is ignored in cross-attention
            is_causal = False
        else:
            T_k = T # Self-attention
            k = self.k_proj(x)
            v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)    # B, H, T, D
        k = k.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T_k, D
        v = v.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T_k, D

        # Apply RoPE if applicable
        if use_rope_for_this_pass:
            cos, sin = self.rotary_emb(T) # Get embeddings for query length T
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            scaling_factor = 1.0 # RoPE often doesn't need explicit scaling
        else:
            scaling_factor = 1.0 / math.sqrt(self.head_dim) # Standard scaling

        # Compute scaled attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling_factor # [B, H, T, T_k]

        # Apply combined masking (padding AND causal)
        final_mask_bool = None # Boolean mask: True indicates position should be masked (-inf)

        # 1. Process padding mask (attn_mask) -> masks Keys/Values
        if attn_mask is not None:
            # Input mask: 1=keep, 0=mask. We need True where mask should be applied.
            if attn_mask.dim() == 2: # Common case: [B, T_k]
                # Expand to broadcast: [B, T_k] -> [B, 1, 1, T_k]
                padding_mask_bool = ~attn_mask.bool().unsqueeze(1).unsqueeze(2)
            elif attn_mask.dim() == 4: # E.g., [B, 1, 1, T_k]
                padding_mask_bool = ~attn_mask.bool()
            else:
                raise ValueError(f"Unsupported attn_mask dimension: {attn_mask.dim()}")
            final_mask_bool = padding_mask_bool # [B, 1, 1, T_k]

        # 2. Process causal mask (if self-attention and is_causal=True) -> masks future Query positions
        if self.is_self_attention and is_causal:
            causal_mask_bool = self._get_causal_mask(T, x.device) # [T, T]
            # Expand to broadcast: [T, T] -> [1, 1, T, T]
            causal_mask_bool = causal_mask_bool.unsqueeze(0).unsqueeze(0)

            if final_mask_bool is not None:
                # Combine: mask if *either* padding mask *or* causal mask applies
                # Broadcasting works: [B, 1, 1, T_k] | [1, 1, T, T] -> [B, 1, T, T] (since T=T_k)
                final_mask_bool = final_mask_bool | causal_mask_bool
            else:
                final_mask_bool = causal_mask_bool # [1, 1, T, T]

        # Apply the combined mask to scores
        if final_mask_bool is not None:
             # Ensure mask shape is compatible with scores [B, H, T, T_k]
             # final_mask_bool is typically [B, 1, T, T_k] or [B, 1, 1, T_k]
             scores = scores.masked_fill(final_mask_bool, torch.finfo(scores.dtype).min)

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v) # [B, H, T, D]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_dropout(self.out_proj(attn_output))


# ==========================================
# 4) Transformer Decoder Block
# ==========================================
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        # Using Pre-LN
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=True, use_rope=True, max_seq_len=max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = 4 * embed_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(), # Consider SwiGLU later
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout) # Dropout for residual connections

    def forward(self, x, attention_mask=None, is_causal=True):
        """
        Args:
            x: Input sequence [B, T, C].
            attention_mask: Padding mask [B, T]. 1 for real tokens, 0 for padding.
            is_causal: Whether the self-attention should be causal.
        """
        # --- Self-Attention (Causal or Bidirectional based on is_causal) ---
        residual = x
        x_norm = self.ln1(x)
        attn_output = self.self_attention(x_norm, attn_mask=attention_mask, is_causal=is_causal)
        x = residual + self.dropout(attn_output)

        # --- Feed-Forward ---
        residual = x
        x_norm = self.ln2(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)

        return x

# ==========================================
# 5) JEPA Predictor Block (Causal Self-Attn, Cross-Attn to Decoder)
# ==========================================
class JEPAPredictorBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        # Pre-LN structure
        self.ln1 = nn.LayerNorm(embed_dim)
        # Causal Self-attention within the predictor (RoPE enabled)
        self.self_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=True, use_rope=True, max_seq_len=max_seq_len)

        self.ln_cross_attn_query = nn.LayerNorm(embed_dim) # LN before cross-attn query input
        self.ln_cross_attn_kv = nn.LayerNorm(embed_dim)    # LN before cross-attn key/value input
        # Cross-attention to backbone decoder output (non-causal, no RoPE)
        self.cross_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=False, use_rope=False, max_seq_len=max_seq_len)

        self.ln3 = nn.LayerNorm(embed_dim)
        hidden_dim = 4 * embed_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, decoder_output, self_attention_mask=None, cross_attention_mask=None):
        """
        Args:
            x: Predictor input sequence [B, T, C].
            decoder_output: Output from the main BackboneDecoder [B, T_kv, C].
            self_attention_mask: Padding mask for predictor input [B, T]. 1=keep, 0=mask.
            cross_attention_mask: Mask for decoder_output (keys/values in cross-attn) [B, T_kv].
                                  Should be JEPA context_mask (1=context, 0=target/pad).
        """
        # --- Causal Self-Attention within Predictor ---
        residual = x
        x_norm = self.ln1(x)
        attn_output = self.self_attention(
            x_norm,
            attn_mask=self_attention_mask, # Use overall padding mask for self-attn
            is_causal=True                 # Self-attention is CAUSAL
        )
        x = residual + self.dropout(attn_output)

        # --- Cross-Attention to Decoder Output ---
        residual = x
        query_norm = self.ln_cross_attn_query(x)           # Normalize query input (from predictor state)
        kv_norm = self.ln_cross_attn_kv(decoder_output)    # Normalize key/value input (from decoder)

        cross_attn_output = self.cross_attention(
            query_norm,                           # Query from predictor
            attn_mask=cross_attention_mask,       # Mask K/V based on JEPA context_mask
            key_value_states=kv_norm              # K/V from (normalized) decoder output
        )
        x = residual + self.dropout(cross_attn_output)

        # --- Feed-Forward ---
        residual = x
        x_norm = self.ln3(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)

        return x

# ==========================================
# 6) Backbone Decoder (Replaces ContextEncoder)
# ==========================================
class BackboneDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, block_size):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.1) # Dropout after embedding

        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, dropout=0.1, max_seq_len=block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights) # Initialize weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias); torch.nn.init.ones_(module.weight)

    def forward(self, x, attention_mask=None, is_causal=True):
        """
        Args:
            x: Input token indices [B, T].
            attention_mask: Padding mask [B, T]. 1=keep, 0=mask.
            is_causal: Controls self-attention masking in DecoderBlocks.
        """
        B, T = x.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        token_emb = self.token_embedding(x) # [B, T, C]
        x = self.dropout(token_emb)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, is_causal=is_causal)

        x = self.ln_f(x)
        return x

# ==========================================
# 7) JEPA Predictor (Using causal self-attn)
# ==========================================
class JEPAPredictor(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, block_size):
        super().__init__()
        self.block_size = block_size
        # Consider using fewer layers for predictor, e.g., predictor_layers = n_layers // 2
        predictor_layers = n_layers # Keep same depth for now

        # Learnable mask token embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # Predictor blocks (using JEPAPredictorBlock)
        self.blocks = nn.ModuleList([
            JEPAPredictorBlock(embed_dim, n_heads, max_seq_len=block_size)
            for _ in range(predictor_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim) # Final layer norm
        self.apply(self._init_weights) # Initialize weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            pass # Only mask_token is an embedding here, initialized separately
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias); torch.nn.init.ones_(module.weight)

    def forward(self, decoder_output_causal, target_spans_indices, context_mask, attention_mask):
        """
        Predict target span representations.

        Args:
            decoder_output_causal: [B, T, C] Embeddings from CAUSAL BackboneDecoder pass.
            target_spans_indices: List[List[Tuple[int, int]]] Target span indices.
            context_mask: [B, T] JEPA context mask (1=context, 0=target/padding).
            attention_mask: [B, T] Overall padding mask (1=real, 0=pad).

        Returns:
            List[List[Tensor]]: Predicted embeddings for target spans per batch item.
        """
        B, T, C = decoder_output_causal.size()

        # Initialize predictor input:
        # Use mask tokens for target positions, causal decoder output for context positions.
        predictor_input = torch.zeros_like(decoder_output_causal)
        mask_token_expanded = self.mask_token.expand(B, T, C)

        # Boolean masks for indexing
        is_context = context_mask.bool()            # Where JEPA context mask is 1
        is_target = (~is_context) & attention_mask.bool() # Where context is 0 AND not padding

        # Populate predictor input
        predictor_input[is_context] = decoder_output_causal[is_context]
        predictor_input[is_target] = mask_token_expanded[is_target]
        # Padding positions remain zero

        # Process through predictor blocks
        x = predictor_input
        for block in self.blocks:
            x = block(
                x,
                decoder_output=decoder_output_causal, # K/V for cross-attention comes from causal decoder
                self_attention_mask=attention_mask,   # Padding mask for predictor's CAUSAL self-attention
                cross_attention_mask=context_mask     # JEPA context mask to select K/V in cross-attention
            )

        x = self.ln_f(x) # Final normalization [B, T, C]

        # Extract predicted embeddings only for the target spans
        predicted_spans = []
        for b in range(B):
            batch_spans = []
            if not target_spans_indices[b]: # Handle cases where no spans were generated for this item
                predicted_spans.append(batch_spans)
                continue
            for start, end in target_spans_indices[b]:
                valid_end = min(end, T) # Ensure end index is within bounds
                if start < valid_end: # Ensure span has non-zero length
                    span_emb = x[b, start:valid_end] # Extract embeddings [SpanLen, C]
                    batch_spans.append(span_emb)
            predicted_spans.append(batch_spans)

        return predicted_spans

# ==========================================
# 8) Target Encoder (EMA copy of BackboneDecoder, runs NON-CAUSALLY)
# ==========================================
class TargetEncoder(nn.Module):
    def __init__(self, backbone_decoder, ema_decay=0.999):
        super().__init__()
        # Create a deep copy of the backbone decoder structure
        self.encoder = copy.deepcopy(backbone_decoder)
        self.ema_decay = ema_decay
        # Disable gradient computation for the target encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_ema(self, backbone_decoder, decay_rate=None):
        """Update target encoder weights using exponential moving average"""
        decay_rate = decay_rate if decay_rate is not None else self.ema_decay
        self.encoder.eval() # Ensure target is in eval mode
        backbone_decoder.eval() # Ensure source is also in eval mode for consistency

        source_params = dict(backbone_decoder.named_parameters())
        target_params = dict(self.encoder.named_parameters())
        assert source_params.keys() == target_params.keys(), "Parameter mismatch between backbone and target encoders!"

        for name, source_param in source_params.items():
            target_params[name].data.mul_(decay_rate).add_(source_param.data, alpha=1 - decay_rate)

    @torch.no_grad()
    def forward(self, x, attention_mask=None):
        """Forward pass for target encoder - runs NON-CAUSALLY"""
        self.encoder.eval() # Ensure target encoder is always in eval mode
        # Call the underlying decoder's forward pass, forcing is_causal=False
        return self.encoder(x, attention_mask=attention_mask, is_causal=False)

# ==========================================
# 9) Complete T-JEPA Model (Decoder Backbone)
# ==========================================
class TJEPAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, block_size, ema_decay=0.999, lm_loss_weight=0.1, pad_token_id=0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.lm_loss_weight = lm_loss_weight
        self.block_size = block_size # Store block_size

        # Main Backbone: Transformer Decoder
        self.decoder_backbone = BackboneDecoder(vocab_size, embed_dim, n_heads, n_layers, block_size)

        # JEPA Predictor
        self.predictor = JEPAPredictor(embed_dim, n_heads, n_layers, block_size)

        # Target Encoder (EMA copy, runs non-causally)
        self.target_encoder = TargetEncoder(self.decoder_backbone, ema_decay)
        # Perform initial weight copy after TargetEncoder is created
        self.target_encoder.update_ema(self.decoder_backbone, decay_rate=0.0)

        # LM Head (predicts next token)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying (tie backbone embedding with LM head)
        self.decoder_backbone.token_embedding.weight = self.lm_head.weight

    def forward(self, x, context_mask, target_spans_indices, attention_mask):
        """
        Orchestrates the forward pass for training.

        Args:
            x: [B, T] Input token sequence.
            context_mask: [B, T] JEPA context mask (1=context, 0=target/pad).
            target_spans_indices: List[List[Tuple[int, int]]] Target span indices.
            attention_mask: [B, T] Padding mask (1=real, 0=pad).

        Returns:
            Dictionary containing outputs needed for loss calculation.
        """

        # 1. Causal pass through the main decoder backbone
        # Used for LM loss and as context for the predictor's cross-attention.
        decoder_output_causal = self.decoder_backbone(
            x,
            attention_mask=attention_mask,
            is_causal=True # Standard causal operation
        ) # [B, T, C]

        # 2. Non-causal pass through the target encoder (EMA copy, no gradients)
        # Used to get the target representations for the JEPA loss.
        with torch.no_grad():
            self.target_encoder.eval() # Ensure target is in eval mode
            target_embeddings_full = self.target_encoder(
                x,
                attention_mask=attention_mask
                # Internally calls backbone with is_causal=False
            ) # [B, T, C]

        # 3. Predictor pass
        # Predicts representations for target spans using the causal decoder output
        # as context in cross-attention.
        predicted_spans_embeddings = self.predictor(
            decoder_output_causal=decoder_output_causal, # Context for cross-attention
            target_spans_indices=target_spans_indices,   # Which spans to predict
            context_mask=context_mask,                   # Mask for cross-attention K/V
            attention_mask=attention_mask                # Padding mask for predictor's self-attention
        ) # List[List[Tensor]]

        # 4. Extract actual target embeddings from the NON-CAUSAL target encoder output
        target_spans_embeddings = []
        for b in range(x.size(0)):
            batch_spans = []
            if not target_spans_indices[b]: # Handle empty span list for this batch item
                target_spans_embeddings.append(batch_spans)
                continue
            for start, end in target_spans_indices[b]:
                valid_end = min(end, x.size(1))
                if start < valid_end:
                    # Extract from the full target embeddings (non-causal)
                    span_emb = target_embeddings_full[b, start:valid_end]
                    batch_spans.append(span_emb)
            target_spans_embeddings.append(batch_spans) # List[List[Tensor]]

        # 5. Calculate LM Logits
        # Based on the output of the CAUSAL decoder backbone pass.
        lm_logits = self.lm_head(decoder_output_causal) # [B, T, VocabSize]

        return {
            "predicted_spans_embeddings": predicted_spans_embeddings, # From Predictor
            "target_spans_embeddings": target_spans_embeddings,     # From Target Encoder (non-causal)
            "lm_logits": lm_logits,                                 # From Backbone Decoder (causal)
            "input_sequence": x,                                    # For LM loss calculation
            "attention_mask": attention_mask,                       # For LM loss masking (optional)
        }

    def update_target_encoder(self):
        """Update target encoder weights using EMA"""
        self.target_encoder.update_ema(self.decoder_backbone)

    def compute_loss(self, outputs):
        """Compute combined JEPA (MSE) and LM (CrossEntropy) loss."""
        # --- JEPA MSE Loss ---
        predicted_spans = outputs["predicted_spans_embeddings"]
        target_spans = outputs["target_spans_embeddings"]
        batch_size = len(predicted_spans)
        jepa_losses = []
        num_valid_comparisons = 0 # Track how many span comparisons actually happen

        for b in range(batch_size):
            num_spans_in_batch_item = len(predicted_spans[b])
            # Ensure target list has same length (should always be true if data prep is correct)
            if num_spans_in_batch_item != len(target_spans[b]):
                # print(f"Warning: Mismatch in number of predicted ({num_spans_in_batch_item}) vs target ({len(target_spans[b])}) spans for batch item {b}.")
                continue # Skip this item if lengths mismatch

            if num_spans_in_batch_item == 0:
                continue # Skip if no spans were generated/extracted for this item

            span_losses_for_batch_item = []
            for i in range(num_spans_in_batch_item):
                pred_span = predicted_spans[b][i] # [SpanLen_pred, C]
                target_span = target_spans[b][i]  # [SpanLen_target, C]

                # Ensure spans are not empty and shapes match exactly
                if pred_span.nelement() > 0 and target_span.nelement() > 0 and pred_span.shape == target_span.shape:
                    # Optional: Normalize embeddings before MSE loss
                    # pred_span_norm = F.normalize(pred_span, p=2, dim=-1)
                    # target_span_norm = F.normalize(target_span, p=2, dim=-1)
                    # loss = F.mse_loss(pred_span_norm, target_span_norm)

                    loss = F.mse_loss(pred_span, target_span)
                    span_losses_for_batch_item.append(loss)
                    num_valid_comparisons += 1
                # else:
                    # Optional: Log why a comparison was skipped
                    # if pred_span.nelement() == 0: print(f"Debug: Skipped empty pred span {b},{i}")
                    # elif target_span.nelement() == 0: print(f"Debug: Skipped empty target span {b},{i}")
                    # else: print(f"Debug: Skipped shape mismatch {pred_span.shape} vs {target_span.shape} for {b},{i}")


            # Average loss across valid spans for this batch item
            if span_losses_for_batch_item:
                 jepa_losses.append(torch.stack(span_losses_for_batch_item).mean())

        # Average JEPA loss over the batch items that had valid spans
        if jepa_losses:
            final_jepa_loss = torch.stack(jepa_losses).mean()
        else:
            # Return zero loss if NO valid spans were compared across the entire batch
            example_tensor = outputs["lm_logits"] # Get device/dtype hint
            final_jepa_loss = torch.tensor(0.0, device=example_tensor.device, dtype=example_tensor.dtype)
            # if num_valid_comparisons == 0: print("Warning: JEPA loss is 0.0 because no valid span comparisons occurred in this batch.")


        # --- LM Cross Entropy Loss ---
        lm_logits = outputs["lm_logits"] # [B, T, V]
        input_sequence = outputs["input_sequence"] # [B, T]

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous() # [B, T-1, V]
        shift_labels = input_sequence[:, 1:].contiguous() # [B, T-1]

        # Flatten the tokens for CrossEntropyLoss
        shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # [B*(T-1), V]
        shift_labels = shift_labels.view(-1) # [B*(T-1)]

        # Calculate loss, ignoring padding tokens
        lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=self.pad_token_id)

        # --- Combine Losses ---
        total_loss = final_jepa_loss + self.lm_loss_weight * lm_loss

        return {
            "total_loss": total_loss,
            "jepa_loss": final_jepa_loss, # This should now be non-zero if spans are generated
            "lm_loss": lm_loss
        }

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_p=0.9):
        """Generate text autoregressively using the BackboneDecoder."""
        self.eval() # Ensure model is in evaluation mode
        B = x.size(0)
        pad_token_id = self.pad_token_id

        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]
            seq_len = x_cond.size(1)

            # Create attention mask for padding (1 for real tokens, 0 for padding)
            attention_mask = (x_cond != pad_token_id).float() # [B, T]

            # Get embeddings from the decoder backbone (CAUSALLY)
            decoder_output = self.decoder_backbone(
                x_cond,
                attention_mask=attention_mask,
                is_causal=True # Explicitly causal for generation
            ) # [B, T, C]

            # Get logits for the *next* token prediction (using the last token's embedding)
            # Apply LM head to the embedding of the very last token in the sequence
            logits = self.lm_head(decoder_output[:, -1, :])  # [B, C] -> [B, V]

            # Apply temperature scaling
            if temperature > 0 and temperature != 1.0:
                 logits = logits / temperature

            # Apply top-p (nucleus) sampling
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter mask back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                # Apply mask by setting logits to -infinity
                logits[indices_to_remove] = float('-inf')

            # Sample from the potentially modified distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # [B, 1]

            # Append sampled token to the running sequence
            x = torch.cat([x, next_token], dim=1)

            # Optional: Check if EOS token was generated in *all* sequences (for batch generation)
            # if hyperparams['eos_token'] is not None and (next_token == hyperparams['eos_token']).all():
            #     break

        return x


# ==========================================
# 10) Evaluation Function
# ==========================================
@torch.no_grad()
def estimate_loss(model, train_df, val_df, hyperparams, device):
    """Estimates loss on train and validation splits."""
    out = {}
    model.eval() # Set model to evaluation mode

    for split, df in [('train', train_df), ('val', val_df)]:
        total_losses = torch.zeros(hyperparams['eval_iters'])
        jepa_losses = torch.zeros(hyperparams['eval_iters'])
        lm_losses = torch.zeros(hyperparams['eval_iters'])

        # Use tqdm for eval iterations if desired, but can be removed
        # pbar_eval = tqdm(range(hyperparams['eval_iters']), desc=f"Eval {split}", leave=False)
        # for k in pbar_eval:
        for k in range(hyperparams['eval_iters']):
            # Get a batch of data
            x, context_mask, target_spans_indices, attention_mask = prepare_batches_from_gsm8k(
                df, hyperparams, device
            )

            # Forward pass through the model
            outputs = model(x, context_mask, target_spans_indices, attention_mask)

            # Compute loss using the model's loss function
            loss_dict = model.compute_loss(outputs)

            # Store losses
            total_losses[k] = loss_dict['total_loss'].item()
            jepa_losses[k] = loss_dict['jepa_loss'].item()
            lm_losses[k] = loss_dict['lm_loss'].item()

        # Calculate average losses for the split
        out[split + '_total'] = total_losses.mean()
        out[split + '_jepa'] = jepa_losses.mean()
        out[split + '_lm'] = lm_losses.mean()

    # model.train() # Caller should reset mode after evaluation
    return out


# ==========================================
# 11) Generate Text Function (Uses model.generate)
# ==========================================
@torch.no_grad()
def generate_from_prompt(model, hyperparams, prompt_text=None, max_new_tokens=200, top_p=None, device="cuda"):
    """Generates text from a prompt using the model's generate method."""
    model.eval() # Ensure evaluation mode
    prompt_text = prompt_text if prompt_text is not None else hyperparams['start_prompt']
    top_p = top_p if top_p is not None else hyperparams['top_p']
    system_prompt = hyperparams['system_prompt']
    full_prompt = f"{system_prompt}\n\nProblem: {prompt_text}\n\n<think>" # Start generation within think tags

    # Encode prompt
    bos_token = hyperparams['bos_token']
    prompt_bytes = [bos_token] + [b for b in full_prompt.encode('utf-8', errors='replace')]
    context = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0) # [1, T_prompt]

    # Use the model's generate method
    full_output_tokens = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=0.8 # Example temperature, could be hyperparameter
    ) # [1, T_prompt + T_new]

    # Decode the full output sequence
    full_output_list = full_output_tokens[0].tolist()

    # Decode bytes to text, handling potential errors and special tokens
    try:
        # Find EOS token if present and truncate
        eos_token = hyperparams['eos_token']
        eos_pos = full_output_list.index(eos_token) if eos_token in full_output_list else -1
        if eos_pos != -1:
            full_output_list = full_output_list[:eos_pos] # Truncate at EOS

        # Remove padding tokens and decode
        pad_token = hyperparams['pad_token']
        decoded_bytes = bytes([tok for tok in full_output_list if tok != pad_token])
        generated_text = decoded_bytes.decode('utf-8', errors='replace')
        return generated_text
    except Exception as e:
        print(f"Decoding error during generation: {e}")
        # Fallback: return raw bytes representation
        return str(bytes(full_output_list))


# ==========================================
# 12) Token-by-Token Generation (Manual loop)
# ==========================================
@torch.no_grad()
def generate_token_by_token(model, hyperparams, prompt_text, max_new_tokens=200, device="cuda"):
    """Generates token by token, printing output, using the decoder model."""
    model.eval() # Ensure evaluation mode
    system_prompt = hyperparams['system_prompt']
    full_prompt = f"{system_prompt}\n\nProblem: {prompt_text}\n\n<think>"
    bos_token = hyperparams['bos_token']
    pad_token = hyperparams['pad_token']
    eos_token = hyperparams['eos_token']

    # Encode prompt
    prompt_bytes = [bos_token] + [b for b in full_prompt.encode('utf-8', errors='replace')]
    context = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0) # [1, T_prompt]

    print(f"\n--- Generating from prompt ---\n{full_prompt}", end="", flush=True)

    generated_tokens = []
    current_byte_fragment = b''

    # Manually loop for token-by-token generation
    for _ in range(max_new_tokens):
        # --- Prepare input for this step ---
        # Crop context if it exceeds block size
        context_cond = context if context.size(1) <= model.block_size else context[:, -model.block_size:]
        # Create attention mask for padding
        attention_mask = (context_cond != pad_token).float() # [1, T_cond]

        # --- Forward pass (Causal) ---
        decoder_output = model.decoder_backbone(
            context_cond,
            attention_mask=attention_mask,
            is_causal=True
        ) # [1, T_cond, C]
        # Get logits for the next token prediction (using the last token's output)
        logits = model.lm_head(decoder_output[:, -1, :]) # [1, V]

        # --- Sampling (Top-p) ---
        top_p = hyperparams['top_p']
        temperature = 0.8 # Example temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # [1, 1]

        # --- Update context and decode/print ---
        next_token_value = next_token.item()
        context = torch.cat([context, next_token], dim=1) # Append to context for next step
        generated_tokens.append(next_token_value)

        # Attempt to decode and print the new byte(s)
        current_byte_fragment += bytes([next_token_value])
        try:
            next_char = current_byte_fragment.decode('utf-8')
            print(next_char, end="", flush=True)
            current_byte_fragment = b'' # Reset fragment if decode succeeds
            time.sleep(0.01) # Small delay for visualization
        except UnicodeDecodeError:
            # If we can't decode (partial UTF-8 character), wait for more bytes
            if len(current_byte_fragment) > 3: # Avoid getting stuck on invalid sequences
                 print("<?>", end="", flush=True) # Print placeholder for invalid sequence
                 current_byte_fragment = b'' # Reset

        # Check for EOS token
        if next_token_value == eos_token:
            print("<EOS>", end="", flush=True)
            break

    print("\n\n--- Generation completed ---")

    # Return the full generated text (including prompt) after loop finishes
    full_generated_list = prompt_bytes + generated_tokens
    try:
        eos_pos = full_generated_list.index(eos_token) if eos_token in full_generated_list else -1
        if eos_pos != -1: full_generated_list = full_generated_list[:eos_pos]
        decoded_bytes = bytes([tok for tok in full_generated_list if tok != pad_token])
        return decoded_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Final decoding error after token-by-token generation: {e}")
        return str(bytes(full_generated_list))


# ==========================================
# 13) Training Implementation
# ==========================================
def train(continue_training=True):
    """Train the T-JEPA DECODER model on GSM8K."""
    # --- Setup ---
    hyperparams = get_hyperparams()
    device = get_device()
    train_df, val_df, test_df = load_gsm8k_data()

    # --- Model Initialization ---
    model = TJEPAModel(
        vocab_size=hyperparams['vocab_size'],
        embed_dim=hyperparams['embed_dim'],
        n_heads=hyperparams['n_heads'],
        n_layers=hyperparams['n_layers'],
        block_size=hyperparams['block_size'],
        ema_decay=hyperparams['ema_decay'],
        lm_loss_weight=hyperparams['lm_loss_weight'],
        pad_token_id=hyperparams['pad_token']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Block Size: {hyperparams['block_size']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}") # Includes decoder, predictor, LM head

    # --- Optimizer ---
    # Filter out parameters that don't require gradients (target encoder)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, # Initial LR, scheduler will adjust
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_loss = float('inf')
    current_step = 0
    checkpoint_path = hyperparams['checkpoint_path']

    if continue_training and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_state = checkpoint['model_state']

            # --- Flexible State Dict Loading ---
            # Handles potential renames (e.g., encoder -> decoder) or minor architecture changes
            current_model_dict = model.state_dict()
            processed_state_dict = {}
            warned_keys = set()
            for k, v in model_state.items():
                new_k = k
                # Example rename: if checkpoint has 'context_encoder', map to 'decoder_backbone'
                if k.startswith("context_encoder."):
                    new_k = k.replace("context_encoder.", "decoder_backbone.", 1)

                if new_k in current_model_dict:
                    if v.shape == current_model_dict[new_k].shape:
                        processed_state_dict[new_k] = v
                    else:
                        if new_k not in warned_keys:
                            print(f"Warning: Shape mismatch for key '{new_k}'. Checkpoint: {v.shape}, Model: {current_model_dict[new_k].shape}. Skipping.")
                            warned_keys.add(new_k)
                # else:
                #     if k not in warned_keys and new_k not in warned_keys: # Avoid double warning if rename failed
                #          print(f"Warning: Key '{k}' (mapped to '{new_k}') not found in current model. Skipping.")
                #          warned_keys.add(k); warned_keys.add(new_k)

            missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)
            if missing_keys: print(f"Warning: Missing keys in final state_dict load: {missing_keys}")
            if unexpected_keys: print(f"Warning: Unexpected keys in final state_dict load: {unexpected_keys}")
            # --- End Flexible State Dict Loading ---


            # Load optimizer state cautiously
            try:
                # Basic check: does the number of parameter groups match?
                if len(optimizer.param_groups) == len(checkpoint['optimizer_state']['param_groups']):
                    # More thorough check: do parameter IDs seem to align? (Heuristic)
                    # This is complex; often safer to reinitialize if model structure changed significantly.
                    # For simplicity, we'll try loading if group count matches.
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print("Optimizer state loaded.")
                else:
                     print("Warning: Optimizer parameter group mismatch. Reinitializing optimizer.")
                     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
            except Exception as e_optim:
                 print(f"Warning: Could not load optimizer state: {e_optim}. Reinitializing.")
                 optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

            # Load training progress
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            current_step = checkpoint.get('current_step', start_epoch * hyperparams['steps_per_epoch'])
            print(f"Resuming from epoch {start_epoch}, step {current_step}.")

            # IMPORTANT: Re-sync target encoder from the loaded backbone weights
            model.target_encoder.update_ema(model.decoder_backbone, decay_rate=0.0)
            print("Target encoder re-synced from loaded backbone weights.")

        except Exception as e:
            print(f"Error loading checkpoint comprehensively: {e}")
            print("Starting training from scratch or with partially loaded state.")
            start_epoch = 0; best_val_loss = float('inf'); current_step = 0
            # Ensure target encoder is initialized correctly if loading failed
            model.target_encoder.update_ema(model.decoder_backbone, decay_rate=0.0)

    else:
        print("Starting training from scratch.")
        # Initial sync of target encoder
        model.target_encoder.update_ema(model.decoder_backbone, decay_rate=0.0)

    # --- LR Scheduler ---
    grad_clip = 1.0
    total_steps = hyperparams['num_epochs'] * hyperparams['steps_per_epoch']
    warmup_steps = 2000 # Example warmup steps
    base_lr = 3e-4
    min_lr = 1e-5

    def get_lr(step):
        # Cosine decay with warmup
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        decay_steps = total_steps - warmup_steps
        steps_after_warmup = step - warmup_steps
        if steps_after_warmup >= decay_steps: # Avoid going past total steps
            return min_lr
        cosine_decay = 0.5 * (1 + math.cos(math.pi * steps_after_warmup / decay_steps))
        decayed_lr = min_lr + (base_lr - min_lr) * cosine_decay
        return max(min_lr, decayed_lr) # Ensure LR doesn't drop below min_lr

    # --- Training Loop ---
    print(f"Starting training on GSM8K dataset with T-JEPA (DECODER bs={hyperparams['block_size']} + RoPE + MTL)...")
    accumulation_steps = hyperparams['accumulation_steps']
    # Optional: Setup Mixed Precision
    # scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    for epoch in range(start_epoch, hyperparams['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{hyperparams['num_epochs']} ---")
        model.train() # Set model to training mode
        epoch_total_loss, epoch_jepa_loss, epoch_lm_loss = 0.0, 0.0, 0.0
        steps_in_epoch = hyperparams['steps_per_epoch']
        optimizer.zero_grad() # Zero gradients at the start of epoch / after optimizer step

        pbar = tqdm(range(steps_in_epoch), desc=f"Epoch {epoch+1}")
        for step_in_epoch in pbar:
            global_step = current_step

            # --- Periodic Evaluation ---
            if global_step > 0 and global_step % hyperparams['eval_interval'] == 0:
                model.eval() # Switch to eval mode
                losses = estimate_loss(model, train_df, val_df, hyperparams, device)
                print(f"\nStep {global_step} Eval:")
                print(f"  Train Total: {losses['train_total']:.4f}, JEPA: {losses['train_jepa']:.4f}, LM: {losses['train_lm']:.4f}")
                print(f"  Val Total:   {losses['val_total']:.4f}, JEPA: {losses['val_jepa']:.4f}, LM: {losses['val_lm']:.4f}")
                model.train() # Switch back to train mode

                # Save best model based on validation total loss
                current_val_loss = losses['val_total']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    save_path = checkpoint_path.replace('.pt', '_best.pt')
                    # Save model state, optimizer, epoch, step, loss
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch,
                        'current_step': global_step,
                        'val_loss': best_val_loss,
                        'hyperparams': hyperparams # Save hyperparams used for this checkpoint
                    }, save_path)
                    print(f"  New best model saved to {save_path}! Val loss: {best_val_loss:.4f}")

            # --- Data Batch ---
            try:
                 x, context_mask, target_spans_indices, attention_mask = prepare_batches_from_gsm8k(
                    train_df, hyperparams, device)
            except Exception as data_err:
                print(f"\nError preparing batch at step {global_step}: {data_err}. Skipping step.")
                # Skip optimizer step if accumulation would be incomplete
                if (step_in_epoch + 1) % accumulation_steps == 0:
                     optimizer.zero_grad() # Reset grads if skipping step at accumulation boundary
                current_step += 1 # Still increment step counter
                continue

            # --- Forward and Loss Calculation ---
            # Optional: Use autocast for mixed precision
            # with torch.autocast(device_type=device if device != 'cpu' else 'cpu', dtype=torch.bfloat16 if device=='cuda' else torch.float32, enabled=(device=='cuda')):
            outputs = model(x, context_mask, target_spans_indices, attention_mask)
            loss_dict = model.compute_loss(outputs)
            total_loss = loss_dict['total_loss']
            jepa_loss = loss_dict['jepa_loss']
            lm_loss = loss_dict['lm_loss']

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / accumulation_steps

            # --- Backward Pass ---
            # scaler.scale(scaled_loss).backward() # With AMP
            scaled_loss.backward() # Without AMP

            # Accumulate epoch losses for monitoring (average per step)
            epoch_total_loss += total_loss.item()
            epoch_jepa_loss += jepa_loss.item()
            epoch_lm_loss += lm_loss.item()

            # --- Optimizer Step (after accumulation) ---
            if (step_in_epoch + 1) % accumulation_steps == 0:
                # Unscale gradients before clipping (required for AMP)
                # scaler.unscale_(optimizer) # With AMP

                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)

                 # Check for NaN/Inf gradients *before* optimizer step
                found_nan_inf = False
                for p in filter(lambda p: p.requires_grad and p.grad is not None, model.parameters()):
                    if not torch.isfinite(p.grad).all():
                        print(f"\nWarning: NaN or Inf found in gradients at step {global_step}. Zeroing gradients for this step.")
                        found_nan_inf = True
                        break
                if found_nan_inf:
                    optimizer.zero_grad() # Skip update if grads are invalid
                else:
                    # Update learning rate based on global step
                    lr = get_lr(global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # Perform optimizer step
                    # scaler.step(optimizer) # With AMP
                    optimizer.step() # Without AMP

                # Update target encoder using EMA after successful optimizer step
                if not found_nan_inf:
                     model.update_target_encoder()

                # Update scaler for next iteration (AMP)
                # scaler.update() # With AMP

                # Zero gradients for the next accumulation cycle
                optimizer.zero_grad()

                # --- Logging ---
                # Calculate average loss over steps completed so far in the epoch
                avg_total_loss = epoch_total_loss / (step_in_epoch + 1)
                avg_jepa_loss = epoch_jepa_loss / (step_in_epoch + 1)
                avg_lm_loss = epoch_lm_loss / (step_in_epoch + 1)
                # Update tqdm progress bar
                pbar.set_description(f"E{epoch+1}, S{global_step+1}/{total_steps}, LR: {lr:.6f}")
                pbar.set_postfix({
                    "AvgLoss": f"{avg_total_loss:.4f}",
                    "JEPA": f"{avg_jepa_loss:.4f}", # Should be non-zero now
                    "LM": f"{avg_lm_loss:.4f}",
                    "LastJEPA": f"{jepa_loss.item():.4f}", # Show last step's JEPA loss
                })

            current_step += 1 # Increment global step counter

        # --- End of Epoch ---
        # Generate sample text
        try:
            print("\nGenerating sample text at end of epoch...")
            model.eval() # Set to eval mode for generation
            sample_text = generate_from_prompt(
                model, hyperparams, hyperparams['start_prompt'],
                max_new_tokens=256, # Shorter sample for epoch end
                device=device
            )
            print(f"Sample: {sample_text}\n" + "-"*20)
            model.train() # Set back to train mode
        except Exception as e:
            print(f"Error generating sample: {e}")
            model.train() # Ensure model is back in train mode

        # Save end-of-epoch checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'current_step': current_step,
            'val_loss': best_val_loss, # Save the best validation loss seen so far
            'hyperparams': hyperparams # Save hyperparams with checkpoint
        }, checkpoint_path)
        print(f"Checkpoint saved at end of epoch {epoch+1} to {checkpoint_path}.")

    print("Training complete!")


# ==========================================
# 14) Inference Implementation
# ==========================================
def inference(model_path, prompt_text, hyperparams_override=None):
    """Run inference with trained DECODER model."""
    device = get_device()

    # --- Load Checkpoint and Hyperparameters ---
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return None

    print(f"Loading model checkpoint from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Load hyperparams from checkpoint if available, otherwise use defaults/overrides
        hyperparams_loaded = checkpoint.get('hyperparams', None)
        if hyperparams_loaded:
            print("Using hyperparameters loaded from checkpoint.")
            hyperparams = hyperparams_loaded
        else:
            print("Warning: Hyperparameters not found in checkpoint, using default values.")
            hyperparams = get_hyperparams()

        # Allow overriding specific hyperparameters for inference
        if hyperparams_override:
            print(f"Applying hyperparameter overrides: {hyperparams_override}")
            hyperparams.update(hyperparams_override)

        print(f"Using hyperparameters for inference: {hyperparams}") # Log effective hyperparams

    except Exception as e:
        print(f"Error loading checkpoint structure: {e}")
        return None

    # --- Create Model Structure based on loaded/effective Hyperparameters ---
    try:
        model = TJEPAModel(
            vocab_size=hyperparams['vocab_size'], embed_dim=hyperparams['embed_dim'],
            n_heads=hyperparams['n_heads'], n_layers=hyperparams['n_layers'],
            block_size=hyperparams['block_size'], ema_decay=hyperparams['ema_decay'], # Needed for structure
            lm_loss_weight=hyperparams['lm_loss_weight'], pad_token_id=hyperparams['pad_token']
        ).to(device)
    except KeyError as e:
         print(f"Error: Missing hyperparameter '{e}' needed to build the model structure.")
         return None

    # --- Load Model State ---
    try:
        model_state = checkpoint['model_state']
        # Flexible loading (handle potential renames/missing/unexpected keys)
        current_model_dict = model.state_dict()
        processed_state_dict = {}
        for k, v in model_state.items():
            new_k = k
            if k.startswith("context_encoder."): new_k = k.replace("context_encoder.", "decoder_backbone.", 1)
            if new_k in current_model_dict and v.shape == current_model_dict[new_k].shape:
                processed_state_dict[new_k] = v
        missing, unexpected = model.load_state_dict(processed_state_dict, strict=False)
        if missing: print(f"  Info: Missing keys while loading state_dict: {missing}")
        if unexpected: print(f"  Info: Unexpected keys while loading state_dict: {unexpected}")
        print("Model state loaded successfully.")
        loaded_epoch = checkpoint.get('epoch', -1); loaded_step = checkpoint.get('current_step', -1)
        print(f"  Checkpoint details: Epoch {loaded_epoch}, Step {loaded_step}, Val Loss {checkpoint.get('val_loss', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error loading model state weights: {e}")
        print("Attempting inference with initialized model weights (may perform poorly).")

    # --- Run Generation ---
    model.eval() # Set to evaluation mode
    print(f"\n--- Generating response for prompt ---")
    print(f"Prompt: {prompt_text}")

    # Use token-by-token generation for streaming output
    result = generate_token_by_token(
        model, hyperparams, prompt_text=prompt_text,
        max_new_tokens=hyperparams.get('generate_num_tokens', 1024), # Use hyperparam, default 512
        device=device
    )
    # Result is printed during generation

    return result

# ==========================================
# 15) Main Entry Point
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    # Load default hyperparameters initially to get paths etc.
    default_hyperparams = get_hyperparams()

    # Choose mode: "train" or "inference"
    MODE = "train"
    # MODE = "inference"

    # Set prompt for inference mode
    INFERENCE_PROMPT = "A rectangle has a length of 15 cm and a width of 8 cm. What is its perimeter and area?"
    # Specify model path for inference (usually the best saved model)
    # Use path from default hyperparams, but it might be overridden if loaded from checkpoint in inference mode
    INFERENCE_MODEL_PATH = default_hyperparams['checkpoint_path'].replace('.pt', '_best.pt')
    # --- End Configuration ---


    if MODE == "train":
        print("Starting training...")
        # Pass continue_training flag from default hyperparams
        train(continue_training=default_hyperparams['continue_training'])

    elif MODE == "inference":
        print("Starting inference...")
        # Check if the specified best model path exists, otherwise try the regular checkpoint
        if not os.path.exists(INFERENCE_MODEL_PATH):
             print(f"Warning: Best model path '{INFERENCE_MODEL_PATH}' not found.")
             base_checkpoint_path = default_hyperparams['checkpoint_path']
             if os.path.exists(base_checkpoint_path):
                 print(f"Attempting to use the base checkpoint path: '{base_checkpoint_path}'")
                 INFERENCE_MODEL_PATH = base_checkpoint_path
             else:
                 print(f"Error: Neither best model nor base checkpoint path found ('{base_checkpoint_path}'). Cannot run inference.")
                 exit() # Exit if no model file found

        # Run inference function
        inference(INFERENCE_MODEL_PATH, INFERENCE_PROMPT) # Hyperparams will be loaded from checkpoint

    else:
        print(f"Unknown mode: {MODE}. Choose 'train' or 'inference'.")
