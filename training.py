"""
T-JEPA MTL Decoder RoPE GSM8K

This script implements a Transformer Decoder model trained with a combination of
Joint Embedding Predictive Architecture (JEPA) and standard Language Modeling (LM)
Multi-Task Learning (MTL) objectives. The model uses Rotary Positional Embeddings (RoPE)
and is designed for solving mathematical reasoning problems, specifically trained and
evaluated on the GSM8K dataset.

Core Components:
1.  **Hyperparameters**: Configuration for model architecture, JEPA, training, etc.
2.  **Data Loading**: Loads and preprocesses the GSM8K dataset.
3.  **Batch Preparation**: Creates batches suitable for JEPA and LM training, including
    masking context/target spans.
4.  **Rotary Positional Embedding (RoPE)**: Implements RoPE for incorporating relative
    positional information in attention.
5.  **Improved Attention**: Multi-head attention module incorporating RoPE and causal masking.
6.  **Decoder Block**: Standard Transformer decoder block using pre-LayerNorm,
    self-attention, and feed-forward layers.
7.  **JEPA Predictor Block**: A specialized block for the JEPA predictor, featuring
    causal self-attention and cross-attention to the backbone decoder's output.
8.  **Backbone Decoder**: The main Transformer decoder network that processes input
    sequences causally for LM prediction.
9.  **JEPA Predictor**: Predicts representations of masked target spans using context
    from the backbone decoder.
10. **Target Encoder**: An Exponential Moving Average (EMA) copy of the Backbone Decoder,
    used non-causally to generate target representations for the JEPA loss.
11. **TJEPAModel**: The complete model integrating the backbone, predictor, target encoder,
    and LM head. Handles the combined forward pass and loss calculation.
12. **Training & Evaluation**: Functions for training the model, estimating loss,
    and generating text.
13. **Inference**: Functionality to load a trained model and generate responses to prompts.
"""

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
from typing import Optional, Tuple, List, Dict, Any

# ==========================================
# 1) Hyperparameters
# ==========================================
def get_hyperparams() -> Dict[str, Any]:
    """
    Returns a dictionary containing hyperparameters for the model and training.

    Returns:
        Dict[str, Any]: A dictionary of hyperparameters.
    """
    return {
        # --- Model Parameters ---
        'batch_size': 2,                  # Number of sequences per batch
        'block_size': 1024,               # Maximum sequence length (context window)
        'vocab_size': 256,                # Size of the vocabulary (byte-level)
        'embed_dim': 512,                 # Dimension of token embeddings and hidden states
        'n_heads': 8,                     # Number of attention heads
        'n_layers': 12,                   # Number of decoder blocks in backbone & predictor

        # --- JEPA Parameters ---
        # Note: Ratios are less critical than absolute span lengths with fixed block size.
        # These settings aim for multiple, reasonably sized target spans.
        'context_span_ratio': 0.6,        # (Informational) Approximate ratio intended for context
        'target_span_ratio': 0.2,         # (Informational) Approximate ratio intended for targets
        'num_target_spans': 8,            # Number of target spans to mask and predict per sequence
        'min_span_length': 32,            # Minimum length of a target span

        # --- Training Parameters ---
        'num_epochs': 50,                 # Total number of training epochs
        'steps_per_epoch': 1000,          # Number of training steps (batches) per epoch
        'eval_interval': 200,             # Evaluate model every N steps
        'eval_iters': 100,                # Number of batches to use for evaluation
        'ema_decay': 0.999,               # Decay rate for the Exponential Moving Average of the target encoder
        'accumulation_steps': 8,          # Accumulate gradients over N steps before optimizer update
        'lm_loss_weight': 0.92,           # Weight factor for the Language Modeling loss component
        'learning_rate': 3e-4,            # Base learning rate for AdamW optimizer
        'min_learning_rate': 1e-5,        # Minimum learning rate for cosine decay scheduler
        'warmup_steps': 2000,             # Number of warmup steps for the learning rate scheduler
        'weight_decay': 0.1,              # Weight decay for AdamW optimizer
        'grad_clip': 1.0,                 # Gradient clipping value

        # --- Special Tokens ---
        'bos_token': 254,                 # Beginning of Sequence token ID (byte value)
        'eos_token': 255,                 # End of Sequence token ID (byte value)
        'pad_token': 0,                   # Padding token ID (byte value, often NULL byte)

        # --- Generation Parameters ---
        'generate_num_tokens': 1024,      # Maximum number of new tokens to generate during inference
        'top_p': 0.8,                     # Nucleus sampling probability threshold
        'temperature': 0.8,               # Sampling temperature for generation
        'start_prompt': "Problem: A bakery produces cakes for $10 each. It costs them $5 in ingredients per cake, and they have a fixed overhead of $200 per day. How many cakes do they need to sell each day to make a daily profit of $100?", # Default prompt for generation examples

        # --- Special Tags for GSM8K formatting ---
        'thinking_tag': "<think>",
        'thinking_end_tag': "</think>",
        'answer_tag': "<answer>",
        'answer_end_tag': "</answer>",

        # --- Paths & Modes ---
        'checkpoint_path': "t_jepa_mtl_decoder_rope_bs1024_checkpoint.pt", # File path for saving/loading checkpoints
        'continue_training': True,        # Whether to resume training from the checkpoint if it exists
        'system_prompt': """Consider this math problem. Think step by step and provide your reasoning between <think> </think> tags, then give your final answer between <answer> </answer> tags.""" # System prompt prepended to inputs
    }

# ==========================================
# 1.1) Select device
# ==========================================
def get_device() -> str:
    """
    Selects the appropriate compute device (MPS, CUDA, or CPU) based on availability.

    Returns:
        str: The selected device name ('mps', 'cuda', or 'cpu').
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device

# ==========================================
# 1.2) Data Loading and Preprocessing for GSM8K
# ==========================================
def load_gsm8k_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the GSM8K dataset, attempting various methods (datasets library,
    Hugging Face Hub parquet files, local JSONL files). Splits the training
    data into training and validation sets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for the
        training, validation, and test sets.

    Raises:
        RuntimeError: If the dataset cannot be loaded through any of the
                      attempted methods.
        ImportError: If necessary libraries like 'datasets', 'pyarrow', 'fsspec'
                     are not installed for certain loading methods.
    """
    print("Loading GSM8K dataset...")
    train_df, test_df = None, None

    # Method 1: Try using the 'datasets' library
    try:
        from datasets import load_dataset
        # Specify cache_dir to avoid potential permission issues in default locations
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        os.makedirs(cache_dir, exist_ok=True)
        dataset = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
        train_df = dataset["train"].to_pandas()
        test_df = dataset["test"].to_pandas()
        print("Dataset loaded successfully using 'datasets' library.")
    except ImportError:
        print("Warning: 'datasets' library not found. Trying alternative methods.")
    except Exception as e:
        print(f"Error loading dataset with 'datasets' library: {e}")
        print("Attempting alternative loading methods...")

    # Method 2: Try loading directly from Hugging Face Hub parquet files
    if train_df is None:
        try:
            # Ensure required libraries are installed: pip install pandas pyarrow fsspec aiohttp
            print("Attempting to load from Hugging Face Hub parquet files...")
            splits = {'train': 'main/train-00000-of-00001.parquet',
                      'test': 'main/test-00000-of-00001.parquet'}
            # Use fsspec for remote file system access (hf://)
            train_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
            test_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])
            print("Dataset loaded successfully using parquet files from Hugging Face Hub.")
        except ImportError:
             print("Warning: 'pyarrow' or 'fsspec' not found. Cannot load parquet files. Trying local files.")
        except Exception as e2:
            print(f"Failed to load dataset using parquet from Hub: {e2}")
            print("Attempting to load from local files...")

    # Method 3: Fallback to local JSONL files (adjust paths if needed)
    if train_df is None:
        local_path_train = "./gsm8k_data/train.jsonl" # Example local path
        local_path_test = "./gsm8k_data/test.jsonl"   # Example local path
        if os.path.exists(local_path_train) and os.path.exists(local_path_test):
             print(f"Attempting to load from local JSONL files: {local_path_train}, {local_path_test}")
             try:
                 train_df = pd.read_json(local_path_train, lines=True)
                 test_df = pd.read_json(local_path_test, lines=True)
                 print("Dataset loaded successfully from local JSONL files.")
             except Exception as e3:
                 print(f"Error loading from local JSONL files: {e3}")
        else:
            print(f"Local files not found at {local_path_train} and {local_path_test}.")

    # Final check: Ensure data was loaded
    if train_df is None or test_df is None:
        raise RuntimeError("Unable to load the GSM8K dataset. Please ensure the 'datasets' library is installed, "
                           "you have internet access for Hugging Face Hub, or provide valid local file paths.")

    print(f"Initial training examples loaded: {len(train_df)}")
    print(f"Initial test examples loaded: {len(test_df)}")

    # Split original training data into new train/validation sets
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    print(f"Final training examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print(f"Test examples: {len(test_df)}")

    return train_df, val_df, test_df

# ==========================================
# 1.3) Prepare data for JEPA training
# ==========================================
def prepare_batches_from_gsm8k(data_df: pd.DataFrame, hyperparams: Dict[str, Any], device: str) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]], torch.Tensor]:
    """
    Creates training batches from the GSM8K DataFrame, formatted for JEPA and LM objectives.

    Each batch item includes:
    - The full sequence (prompt + reasoning + answer) encoded as bytes.
    - A context mask indicating which tokens are kept for context (1) and which
      are masked targets for JEPA prediction (0).
    - A list of (start, end) indices for the target spans.
    - An attention mask indicating real tokens (1) vs padding tokens (0).

    Args:
        data_df (pd.DataFrame): DataFrame containing GSM8K data ('question', 'answer').
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters.
        device (str): The compute device ('cuda', 'cpu', 'mps').

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]], torch.Tensor]:
        - x: Batch of input sequences [B, T], dtype=long.
        - context_masks: JEPA context mask [B, T], dtype=float (1=context, 0=target/pad).
        - target_spans_indices: List (length B) of lists of (start, end) tuples for targets.
        - attention_mask: Padding mask [B, T], dtype=float (1=real token, 0=pad).
    """
    # Sample random examples from the dataframe
    batch_indices = torch.randint(0, len(data_df), (hyperparams['batch_size'],))
    batch_examples = data_df.iloc[batch_indices]

    # Get hyperparameters
    block_size = hyperparams['block_size']
    bos_token = hyperparams['bos_token']
    eos_token = hyperparams['eos_token']
    pad_token = hyperparams['pad_token']
    num_target_spans = hyperparams['num_target_spans']
    min_span_length = hyperparams['min_span_length']

    # Initialize lists to store batch data
    full_sequences = []
    context_masks = [] # Mask for JEPA context (1=context, 0=target/padding)
    target_spans_indices = [] # List of (start, end) tuples for each example
    attention_masks_list = [] # List for building the final attention mask tensor

    for _, row in batch_examples.iterrows():
        # --- Format the text ---
        question = row['question']
        answer_reasoning = row['answer'] # The full reasoning text
        system_prompt = hyperparams['system_prompt']
        thinking_tag = hyperparams['thinking_tag']
        thinking_end_tag = hyperparams['thinking_end_tag']
        answer_tag = hyperparams['answer_tag']
        answer_end_tag = hyperparams['answer_end_tag']

        # Extract the final numeric answer from the reasoning text
        # This logic attempts to parse the common "#### <number>" format in GSM8K
        answer_lines = answer_reasoning.strip().split('\n')
        final_answer_line = answer_lines[-1] if answer_lines else ""
        final_answer_numeric = ''.join(filter(lambda x: x.isdigit() or x == '.', final_answer_line.split('####')[-1].strip()))
        if not final_answer_numeric: # Fallback if "####" format not found
             final_answer_numeric = "ERROR" # Or handle more gracefully

        # Construct the full text with prompts and tags
        full_text = f"{system_prompt}\n\nProblem: {question}\n\n{thinking_tag}{answer_reasoning}{thinking_end_tag}\n\n{answer_tag}{final_answer_numeric}{answer_end_tag}"

        # --- Encode, Pad, Truncate ---
        full_bytes = [bos_token] + list(full_text.encode('utf-8', errors='replace')) + [eos_token]
        seq_length_unpadded = len(full_bytes) # Length before padding/truncation

        if seq_length_unpadded > block_size:
            full_bytes = full_bytes[:block_size]
            seq_length = block_size # Actual length used
        else:
            padding_needed = block_size - seq_length_unpadded
            full_bytes = full_bytes + [pad_token] * padding_needed
            seq_length = seq_length_unpadded # Use original length for mask logic

        # --- Create Masks ---
        # Attention Mask: 1 for real tokens (including BOS/EOS), 0 for padding
        current_attention_mask = torch.zeros(block_size, dtype=torch.float)
        current_attention_mask[:seq_length] = 1.0

        # Context Mask (for JEPA): Initialize all real tokens as potential context (1.0)
        current_context_mask = torch.zeros(block_size, dtype=torch.float)
        current_context_mask[:seq_length] = 1.0 # Only real tokens can be context or targets

        # --- Select Target Spans for JEPA ---
        current_target_spans = []
        # Indices of real tokens available for masking (exclude padding)
        # Start after BOS (index 0) and end before potential EOS/padding
        potential_indices = list(range(1, seq_length -1)) # Avoid masking BOS/EOS usually
        available_indices = [idx for idx in potential_indices if idx < seq_length] # Ensure indices are within bounds


        for _ in range(num_target_spans):
            # Check if enough *remaining* tokens are available to form a min_span_length span
            if len(available_indices) < min_span_length:
                break # Not enough remaining candidate tokens

            # Determine maximum possible span length dynamically
            # Limit max span length (e.g., max 20% of sequence or remaining tokens)
            max_possible_len = min(len(available_indices), int(seq_length * 0.2))
            if max_possible_len < min_span_length:
                 continue # Skip if even the max possible length is too small

            # Randomly choose target span length between min and max possible
            # Ensure upper bound is strictly greater than lower bound for randint
            span_length = torch.randint(min_span_length, max(min_span_length + 1, max_possible_len + 1), (1,)).item()

            # Choose random starting position *from the list of currently available indices*
            if len(available_indices) - span_length < 0:
                continue # Safety check, should not happen with max_possible_len logic
            start_idx_in_available = torch.randint(0, len(available_indices) - span_length + 1, (1,)).item()
            start_pos = available_indices[start_idx_in_available] # Actual index in the sequence

            # Calculate end position, ensuring it doesn't exceed sequence length
            end_pos = min(start_pos + span_length, seq_length)
            actual_span_length = end_pos - start_pos

            # Skip if the span ended up being too short (e.g., hit seq_length boundary early)
            if actual_span_length < min_span_length // 2: # Allow slightly shorter if near end
                continue

            # Mark positions in the target span on the context mask (set to 0)
            current_context_mask[start_pos:end_pos] = 0.0

            # Store span positions (start, end)
            current_target_spans.append((start_pos, end_pos))

            # Update available indices: remove indices used by the just-selected target span
            span_indices_set = set(range(start_pos, end_pos))
            available_indices = [idx for idx in available_indices if idx not in span_indices_set]

            # Break if no more indices are available
            if not available_indices:
                break

        # Add processed data for this example to the batch lists
        full_sequences.append(full_bytes)
        context_masks.append(current_context_mask)
        target_spans_indices.append(current_target_spans)
        attention_masks_list.append(current_attention_mask)

    # Convert lists to tensors and move to the specified device
    x = torch.tensor(full_sequences, dtype=torch.long, device=device)
    context_masks_tensor = torch.stack(context_masks).to(device) # [B, T]
    attention_mask_tensor = torch.stack(attention_masks_list).to(device) # [B, T]

    return x, context_masks_tensor, target_spans_indices, attention_mask_tensor

# ==========================================
# 2) Rotary Positional Embedding (RoPE)
# ==========================================
class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE) for incorporating relative
    positional information into transformer attention layers.

    RoPE applies rotations to query and key vectors based on their absolute
    positions, such that the dot product (attention score) depends only on the
    relative positions and the queries/keys themselves.

    Args:
        dim (int): The dimension of the features to be rotated (usually head dimension).
        max_seq_len (int): The maximum sequence length for which to precompute
                           rotary embeddings. Should match or exceed the model's block size.
        base (int): The base value used in the inverse frequency calculation. Default is 10000.
        device (Optional[str]): The device to store the precomputed buffers on.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000, device: Optional[str] = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Calculate inverse frequencies (theta_i in the RoPE paper)
        # Shape: [dim / 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute angles (m * theta_i) for all positions up to max_seq_len
        t = torch.arange(self.max_seq_len, device=device, dtype=self.inv_freq.dtype)
        # freqs shape: [max_seq_len, dim / 2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Concatenate frequencies for both halves of the dimension
        # emb shape: [max_seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Precompute cosine and sine values
        # Stored as buffers, not model parameters
        # cos_cached shape: [max_seq_len, dim]
        # sin_cached shape: [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the precomputed cosine and sine embeddings for a given sequence length.

        Args:
            seq_len (int): The length of the sequence for which to get embeddings.
                           Must be less than or equal to `max_seq_len`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - cos: Cosine values for RoPE [seq_len, dim].
            - sin: Sine values for RoPE [seq_len, dim].

        Raises:
            ValueError: If `seq_len` exceeds `max_seq_len`.
        """
        if seq_len > self.max_seq_len:
            # Dynamic extension is complex; common practice is to set max_seq_len large enough.
            raise ValueError(f"RoPE sequence length {seq_len} exceeds precomputed max {self.max_seq_len}")

        # Return the precomputed values up to the requested sequence length
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.
    Used in applying RoPE. Splits the last dimension, negates the second half,
    and concatenates them in reversed order.

    Args:
        x (torch.Tensor): Input tensor, e.g., query or key. Shape [..., dim].

    Returns:
        torch.Tensor: Tensor with the second half of the last dimension rotated. Shape [..., dim].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embedding to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor. Shape [B, H, T, D_head].
        k (torch.Tensor): Key tensor. Shape [B, H, T_k, D_head].
        cos (torch.Tensor): Precomputed cosine values from RotaryEmbedding. Shape [T, D_head] or [T_k, D_head].
        sin (torch.Tensor): Precomputed sine values from RotaryEmbedding. Shape [T, D_head] or [T_k, D_head].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - q_embed: Query tensor with RoPE applied. Shape [B, H, T, D_head].
        - k_embed: Key tensor with RoPE applied. Shape [B, H, T_k, D_head].
    """
    # Add broadcastable dimensions for batch and head: [T, D_head] -> [1, 1, T, D_head]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation using the RoPE formula:
    # q_rot = q * cos + rotate_half(q) * sin
    # k_rot = k * cos + rotate_half(k) * sin
    # Note: `cos` and `sin` are broadcast across batch and head dimensions.
    # The sequence length dimension (T or T_k) must match between q/k and cos/sin.
    q_embed = (q * cos[:,:,:q.shape[-2],:]) + (rotate_half(q) * sin[:,:,:q.shape[-2],:])
    k_embed = (k * cos[:,:,:k.shape[-2],:]) + (rotate_half(k) * sin[:,:,:k.shape[-2],:])
    return q_embed, k_embed


# ==========================================
# 3) Improved Attention Mechanism (with RoPE and Causal Masking)
# ==========================================
class ImprovedAttention(nn.Module):
    """
    Multi-Head Attention layer with optional RoPE and causal masking.

    Supports both self-attention and cross-attention. Uses RoPE for positional
    encoding in self-attention if enabled. Applies causal masking for
    autoregressive tasks and padding masking to ignore pad tokens.

    Args:
        embed_dim (int): Total dimension of the input embedding.
        n_heads (int): Number of parallel attention heads.
        is_self_attention (bool): True if this is a self-attention layer, False for cross-attention.
                                  RoPE is only applied if True. Default is True.
        use_rope (bool): Whether to apply Rotary Positional Embeddings (only relevant
                         if is_self_attention=True). Default is True.
        max_seq_len (int): Maximum sequence length, used to initialize RoPE. Default is 2048.
        dropout (float): Dropout rate for attention weights and output projection. Default is 0.1.
    """
    def __init__(self, embed_dim: int, n_heads: int, is_self_attention: bool = True, use_rope: bool = True, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by n_heads"
        self.is_self_attention = is_self_attention
        self.use_rope = use_rope

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Instantiate RoPE module if needed (only for self-attention)
        if self.use_rope and self.is_self_attention:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.rotary_emb = None

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Buffer for caching the causal mask to avoid recomputation
        self.register_buffer("causal_mask_cache", None, persistent=False)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Retrieves or creates a causal mask for a given sequence length T.

        The mask is upper triangular (True indicates positions to be masked).

        Args:
            T (int): The sequence length.
            device (torch.device): The device to create the mask on if not cached.

        Returns:
            torch.Tensor: A boolean causal mask of shape [T, T].
        """
        # Check cache first
        if self.causal_mask_cache is None or self.causal_mask_cache.shape[-1] < T:
            # Create lower triangular mask (True for positions to be masked: future tokens)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
            self.causal_mask_cache = mask
        # Return the sub-mask for the current sequence length T, ensuring it's on the correct device
        return self.causal_mask_cache[:T, :T].to(device=device)


    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_value_states: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        """
        Performs the forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Query input tensor. Shape [B, T, C].
            attn_mask (Optional[torch.Tensor]): Padding mask. Indicates which key/value
                                                positions are valid (1) or padding (0).
                                                Shape [B, T_k] or broadcastable (e.g., [B, 1, 1, T_k]).
                                                Defaults to None (no padding mask).
            key_value_states (Optional[torch.Tensor]): Optional key/value input tensor
                                                       for cross-attention. Shape [B, T_k, C].
                                                       If None, performs self-attention using `x`.
                                                       Defaults to None.
            is_causal (bool): If True and this is a self-attention layer, apply a causal
                              mask to prevent attending to future positions.
                              Defaults to False.

        Returns:
            torch.Tensor: The output tensor after attention and projection. Shape [B, T, C].
        """
        B, T, C = x.size() # Batch size, Query sequence length, Embedding dimension
        is_cross_attn = key_value_states is not None
        # Determine if RoPE should be applied in this specific call
        use_rope_for_this_pass = self.use_rope and self.is_self_attention and not is_cross_attn and self.rotary_emb is not None

        # --- 1. Linear Projections ---
        q = self.q_proj(x) # [B, T, C]

        # Project keys and values from `x` (self-attention) or `key_value_states` (cross-attention)
        if is_cross_attn:
            T_k = key_value_states.size(1) # Key/Value sequence length
            k = self.k_proj(key_value_states) # [B, T_k, C]
            v = self.v_proj(key_value_states) # [B, T_k, C]
            is_causal = False # Causal mask is not used in cross-attention
        else:
            T_k = T # Key/Value sequence length is same as Query length for self-attn
            k = self.k_proj(x) # [B, T, C]
            v = self.v_proj(x) # [B, T, C]

        # --- 2. Reshape for Multi-Head Attention ---
        # Reshape from [B, SeqLen, C] to [B, NumHeads, SeqLen, HeadDim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)    # [B, H, T, D_head]
        k = k.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T_k, D_head]
        v = v.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T_k, D_head]

        # --- 3. Apply RoPE (if applicable) ---
        if use_rope_for_this_pass:
            # Get rotary embeddings for the query sequence length T
            cos, sin = self.rotary_emb(T) # Shapes [T, D_head]
            # Apply RoPE to queries and keys
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            scaling_factor = 1.0 # RoPE often doesn't require explicit scaling like 1/sqrt(d_k)
        else:
            # Use standard scaling for dot-product attention
            scaling_factor = 1.0 / math.sqrt(self.head_dim)

        # --- 4. Compute Attention Scores ---
        # q: [B, H, T, D_head], k.transpose: [B, H, D_head, T_k] -> scores: [B, H, T, T_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling_factor

        # --- 5. Apply Masking (Padding and/or Causal) ---
        # Create a combined boolean mask where True indicates positions to be masked (-inf)
        final_mask_bool: Optional[torch.Tensor] = None

        # 5a. Apply Padding Mask (attn_mask affects Keys/Values)
        if attn_mask is not None:
            # Input mask `attn_mask`: 1=keep, 0=mask. We need a boolean mask where True means apply masking.
            if attn_mask.dim() == 2: # Common case: [B, T_k]
                # Expand to broadcast correctly with scores [B, H, T, T_k]
                # -> [B, 1, 1, T_k]
                padding_mask_bool = ~attn_mask.bool().unsqueeze(1).unsqueeze(2)
            elif attn_mask.dim() == 4: # Less common, e.g., already broadcastable [B, 1, 1, T_k]
                padding_mask_bool = ~attn_mask.bool()
            else:
                raise ValueError(f"Unsupported attn_mask dimension: {attn_mask.dim()}. Expected 2 or 4.")
            final_mask_bool = padding_mask_bool # Initialize combined mask

        # 5b. Apply Causal Mask (if self-attention and is_causal=True; affects Query positions)
        if self.is_self_attention and is_causal:
            # Get causal mask for query length T: [T, T] (True means mask)
            causal_mask_bool = self._get_causal_mask(T, x.device)
            # Expand to broadcast correctly with scores [B, H, T, T_k] (Note: T=T_k here)
            # -> [1, 1, T, T]
            causal_mask_bool = causal_mask_bool.unsqueeze(0).unsqueeze(0)

            if final_mask_bool is not None:
                # Combine masks: Mask if *either* padding mask *or* causal mask applies.
                # Broadcasting works: [B, 1, 1, T_k] | [1, 1, T, T] -> [B, 1, T, T]
                final_mask_bool = final_mask_bool | causal_mask_bool
            else:
                # Only causal mask is needed
                final_mask_bool = causal_mask_bool

        # Apply the combined mask to the attention scores
        if final_mask_bool is not None:
             # Fill masked positions with a large negative value before softmax
             scores = scores.masked_fill(final_mask_bool, torch.finfo(scores.dtype).min)

        # --- 6. Apply Softmax and Dropout ---
        # Softmax converts scores to probabilities along the key sequence length dimension (T_k)
        attn_weights = F.softmax(scores, dim=-1) # [B, H, T, T_k]
        attn_weights = self.attn_dropout(attn_weights)

        # --- 7. Apply Attention to Values ---
        # attn_weights: [B, H, T, T_k], v: [B, H, T_k, D_head] -> attn_output: [B, H, T, D_head]
        attn_output = torch.matmul(attn_weights, v)

        # --- 8. Reshape and Project Output ---
        # Concatenate heads: [B, H, T, D_head] -> [B, T, H*D_head] = [B, T, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        # Final linear projection and dropout
        return self.out_dropout(self.out_proj(attn_output)) # [B, T, C]


# ==========================================
# 4) Transformer Decoder Block
# ==========================================
class DecoderBlock(nn.Module):
    """
    A single block of the Transformer Decoder architecture.

    Uses Pre-Layer Normalization (Pre-LN) structure: LN -> Attention -> Residual ->
    LN -> FeedForward -> Residual. Includes dropout on residual connections.

    Args:
        embed_dim (int): Dimension of the input embedding and hidden states.
        n_heads (int): Number of attention heads for the self-attention layer.
        dropout (float): Dropout rate for feed-forward and residual connections. Default is 0.1.
        max_seq_len (int): Maximum sequence length, passed to the attention layer for RoPE.
                           Default is 2048.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        # Layer Normalization before self-attention
        self.ln1 = nn.LayerNorm(embed_dim)
        # Multi-head self-attention with RoPE enabled
        self.self_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=True, use_rope=True, max_seq_len=max_seq_len, dropout=dropout)

        # Layer Normalization before feed-forward network
        self.ln2 = nn.LayerNorm(embed_dim)
        # Feed-forward network (often 4x embedding dim)
        hidden_dim = 4 * embed_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(), # Activation function (Gaussian Error Linear Unit)
            # Consider replacing GELU with SwiGLU for potential performance gains later
            # nn.SiLU(), # Swish/SiLU activation
            # nn.Linear(hidden_dim, hidden_dim), # SwiGLU uses gated linear units
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), # Project back to embedding dimension
            nn.Dropout(dropout)
        )
        # Dropout applied to the output of the attention and feed-forward layers before adding residual
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, is_causal: bool = True) -> torch.Tensor:
        """
        Performs the forward pass of the Decoder block.

        Args:
            x (torch.Tensor): Input sequence tensor. Shape [B, T, C].
            attention_mask (Optional[torch.Tensor]): Padding mask for self-attention.
                                                     Shape [B, T]. 1 for real tokens, 0 for padding.
                                                     Defaults to None.
            is_causal (bool): Whether the self-attention should be causal (masked).
                              Typically True for decoders during training/generation.
                              Defaults to True.

        Returns:
            torch.Tensor: Output tensor after the decoder block. Shape [B, T, C].
        """
        # --- Self-Attention Sub-layer (with Pre-LN and Residual) ---
        residual = x
        # Apply LayerNorm before attention
        x_norm = self.ln1(x)
        # Perform self-attention (causal or bidirectional based on is_causal)
        attn_output = self.self_attention(x_norm, attn_mask=attention_mask, is_causal=is_causal)
        # Add residual connection after dropout
        x = residual + self.dropout(attn_output)

        # --- Feed-Forward Sub-layer (with Pre-LN and Residual) ---
        residual = x
        # Apply LayerNorm before feed-forward
        x_norm = self.ln2(x)
        # Pass through feed-forward network
        ff_output = self.feed_forward(x_norm)
        # Add residual connection after dropout
        x = residual + self.dropout(ff_output)

        return x

# ==========================================
# 5) JEPA Predictor Block (Causal Self-Attn, Cross-Attn to Decoder)
# ==========================================
class JEPAPredictorBlock(nn.Module):
    """
    A block specifically designed for the JEPA Predictor module.

    This block includes:
    1. Causal Self-Attention: Operates on the predictor's internal state, using RoPE.
       Masked by the overall padding mask.
    2. Cross-Attention: Attends to the output of the main Backbone Decoder (causal pass).
       The predictor's state forms the queries, and the backbone output forms keys/values.
       Masked by the JEPA `context_mask` to only attend to context tokens from the backbone.
    3. Feed-Forward Network.

    Uses Pre-LN structure similar to the DecoderBlock.

    Args:
        embed_dim (int): Dimension of the input embedding and hidden states.
        n_heads (int): Number of attention heads for both self- and cross-attention.
        dropout (float): Dropout rate. Default is 0.1.
        max_seq_len (int): Maximum sequence length, passed to attention layers for RoPE.
                           Default is 2048.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        # --- Causal Self-Attention ---
        self.ln1 = nn.LayerNorm(embed_dim) # Pre-LN for self-attention
        # Self-attention within the predictor sequence (always causal, uses RoPE)
        self.self_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=True, use_rope=True, max_seq_len=max_seq_len, dropout=dropout)

        # --- Cross-Attention to Backbone Decoder Output ---
        self.ln_cross_attn_query = nn.LayerNorm(embed_dim) # Pre-LN for cross-attention query input (from predictor state)
        self.ln_cross_attn_kv = nn.LayerNorm(embed_dim)    # Pre-LN for cross-attention key/value input (from backbone)
        # Cross-attention layer (queries from predictor, K/V from backbone, non-causal, no RoPE needed here)
        self.cross_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=False, use_rope=False, max_seq_len=max_seq_len, dropout=dropout)

        # --- Feed-Forward Network ---
        self.ln3 = nn.LayerNorm(embed_dim) # Pre-LN for feed-forward
        hidden_dim = 4 * embed_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                decoder_output: torch.Tensor,
                self_attention_mask: Optional[torch.Tensor] = None,
                cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the JEPA Predictor block.

        Args:
            x (torch.Tensor): Input sequence tensor for the predictor. Shape [B, T, C].
                              Typically contains mask tokens for target spans and
                              backbone embeddings for context spans.
            decoder_output (torch.Tensor): Output from the main BackboneDecoder (causal pass).
                                           Used as Key/Value states in cross-attention.
                                           Shape [B, T_kv, C]. (T_kv is typically T).
            self_attention_mask (Optional[torch.Tensor]): Padding mask for the predictor's
                                                          input `x`. Used in the causal
                                                          self-attention layer. Shape [B, T].
                                                          1=keep, 0=mask. Defaults to None.
            cross_attention_mask (Optional[torch.Tensor]): Mask for the `decoder_output`
                                                            (keys/values in cross-attention).
                                                            This should typically be the JEPA
                                                            `context_mask` (1=context, 0=target/pad)
                                                            to ensure the predictor only attends
                                                            to context positions from the backbone.
                                                            Shape [B, T_kv]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after the predictor block. Shape [B, T, C].
        """
        # --- 1. Causal Self-Attention within Predictor ---
        residual = x
        x_norm = self.ln1(x)
        # Self-attention uses the overall padding mask (`self_attention_mask`) and is always causal.
        attn_output = self.self_attention(
            x_norm,
            attn_mask=self_attention_mask,
            is_causal=True # Predictor's self-attention should look back at its own state
        )
        x = residual + self.dropout(attn_output)

        # --- 2. Cross-Attention to Decoder Output ---
        residual = x
        # Normalize the query (predictor's current state)
        query_norm = self.ln_cross_attn_query(x)
        # Normalize the keys/values (backbone decoder's output)
        kv_norm = self.ln_cross_attn_kv(decoder_output)

        # Perform cross-attention.
        # Query comes from predictor state (`query_norm`).
        # Key/Value come from backbone output (`kv_norm`).
        # `cross_attention_mask` (JEPA context mask) masks the K/V source, allowing attention only to context tokens.
        cross_attn_output = self.cross_attention(
            query_norm,                           # Query [B, T, C]
            attn_mask=cross_attention_mask,       # Mask K/V based on JEPA context [B, T_kv]
            key_value_states=kv_norm              # Key/Value [B, T_kv, C]
            # `is_causal` is implicitly False in cross-attention
        )
        x = residual + self.dropout(cross_attn_output)

        # --- 3. Feed-Forward Network ---
        residual = x
        x_norm = self.ln3(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)

        return x


# ==========================================
# 6) Backbone Decoder (Replaces ContextEncoder)
# ==========================================
class BackboneDecoder(nn.Module):
    """
    The main Transformer Decoder model which serves as the backbone.

    It processes the input sequence token by token. When run causally (`is_causal=True`),
    it generates representations suitable for next-token prediction (LM loss) and
    provides context for the JEPA predictor. When run non-causally (`is_causal=False`),
    as part of the TargetEncoder (EMA copy), it generates representations using
    bidirectional context, suitable for the JEPA target embeddings.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of token embeddings and hidden states.
        n_heads (int): Number of attention heads in each DecoderBlock.
        n_layers (int): Number of DecoderBlocks stacked.
        block_size (int): Maximum sequence length (context window).
        dropout (float): Dropout rate after token embedding. Default is 0.1.
    """
    def __init__(self, vocab_size: int, embed_dim: int, n_heads: int, n_layers: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Dropout applied after embedding lookup
        self.dropout = nn.Dropout(dropout)

        # Stack of DecoderBlocks
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, dropout=dropout, max_seq_len=block_size)
            for _ in range(n_layers)
        ])
        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(embed_dim)

        # Apply custom weight initialization
        self.apply(self._init_weights)
        print(f"Backbone Decoder initialized with {n_layers} layers.")

    def _init_weights(self, module: nn.Module):
        """Initializes weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, is_causal: bool = True) -> torch.Tensor:
        """
        Performs the forward pass of the Backbone Decoder.

        Args:
            x (torch.Tensor): Input token indices. Shape [B, T].
            attention_mask (Optional[torch.Tensor]): Padding mask. Shape [B, T].
                                                     1 for real tokens, 0 for padding.
                                                     Defaults to None.
            is_causal (bool): Controls the self-attention masking in the DecoderBlocks.
                              True for standard causal decoding (LM, predictor context).
                              False for non-causal encoding (JEPA targets via TargetEncoder).
                              Defaults to True.

        Returns:
            torch.Tensor: Output embeddings for the sequence. Shape [B, T, C].

        Raises:
            AssertionError: If the input sequence length `T` exceeds `block_size`.
        """
        B, T = x.size()
        assert T <= self.block_size, f"Input sequence length {T} exceeds model block size {self.block_size}"

        # 1. Get token embeddings
        token_emb = self.token_embedding(x) # [B, T, C]
        # Apply dropout to embeddings
        h = self.dropout(token_emb)

        # 2. Pass through DecoderBlocks
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask, is_causal=is_causal)

        # 3. Apply final Layer Normalization
        h = self.ln_f(h) # [B, T, C]
        return h


# ==========================================
# 7) JEPA Predictor (Using causal self-attn and cross-attn)
# ==========================================
class JEPAPredictor(nn.Module):
    """
    JEPA Predictor module.

    Predicts the representations of masked target spans based on the context
    provided by the Backbone Decoder's causal output. It uses a stack of
    `JEPAPredictorBlock` layers.

    The input to the predictor consists of:
    - Embeddings from the Backbone Decoder for context positions.
    - A learned `mask_token` embedding for target positions.

    Args:
        embed_dim (int): Dimension of embeddings and hidden states.
        n_heads (int): Number of attention heads in each JEPAPredictorBlock.
        n_layers (int): Number of JEPAPredictorBlocks to stack.
                        (Consider using fewer layers than the backbone, e.g., n_layers // 2).
        block_size (int): Maximum sequence length, needed for predictor blocks.
        dropout (float): Dropout rate. Default is 0.1.
    """
    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        # Using the same number of layers as backbone for simplicity, but could be fewer.
        predictor_layers = n_layers
        print(f"JEPA Predictor initialized with {predictor_layers} layers.")

        # Learnable embedding for masked positions ([M] token in papers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Initialize mask token embedding
        torch.nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # Stack of JEPAPredictorBlocks
        self.blocks = nn.ModuleList([
            JEPAPredictorBlock(embed_dim, n_heads, dropout=dropout, max_seq_len=block_size)
            for _ in range(predictor_layers)
        ])
        # Final Layer Normalization for the predictor output
        self.ln_f = nn.LayerNorm(embed_dim)

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights for linear layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            pass # Only mask_token is Parameter here, initialized in __init__
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self,
                decoder_output_causal: torch.Tensor,
                target_spans_indices: List[List[Tuple[int, int]]],
                context_mask: torch.Tensor,
                attention_mask: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Performs the forward pass of the JEPA Predictor.

        Args:
            decoder_output_causal (torch.Tensor): Embeddings from the CAUSAL pass
                                                  of the BackboneDecoder. Shape [B, T, C].
                                                  Used as context input and for cross-attention K/V.
            target_spans_indices (List[List[Tuple[int, int]]]): List (length B) containing
                                                                 lists of (start, end) indices
                                                                 for target spans in each sequence.
            context_mask (torch.Tensor): JEPA context mask. Shape [B, T].
                                         1 indicates context, 0 indicates target or padding.
                                         Used to select K/V in cross-attention and build predictor input.
            attention_mask (torch.Tensor): Overall padding mask. Shape [B, T].
                                           1 indicates real tokens, 0 indicates padding.
                                           Used for masking in the predictor's self-attention.

        Returns:
            List[List[torch.Tensor]]: A list (length B) where each element is a list of
                                      predicted embedding tensors for the target spans
                                      in that sequence. Each inner tensor has shape
                                      [SpanLen, C]. Returns empty inner lists if no
                                      valid spans were processed for a batch item.
        """
        B, T, C = decoder_output_causal.size()

        # --- 1. Prepare Predictor Input ---
        # Initialize predictor input tensor with zeros
        predictor_input = torch.zeros_like(decoder_output_causal)
        # Expand mask token to match batch and sequence dimensions for easy assignment
        mask_token_expanded = self.mask_token.expand(B, T, C)

        # Create boolean masks for efficient indexing
        # `is_context`: Where JEPA context mask is 1 (these positions get backbone output)
        is_context = context_mask.bool()
        # `is_target`: Where JEPA context mask is 0 AND it's not a padding token
        # These positions get the learned mask_token embedding.
        is_target = (~is_context) & attention_mask.bool()

        # Populate the predictor input tensor:
        # - Copy backbone embeddings for context positions
        predictor_input[is_context] = decoder_output_causal[is_context]
        # - Fill target positions with the learned mask token
        predictor_input[is_target] = mask_token_expanded[is_target]
        # Padding positions remain zeros (masked out by attention_mask later)

        # --- 2. Process through Predictor Blocks ---
        # The predictor input `h` evolves through the blocks
        h = predictor_input
        for block in self.blocks:
            h = block(
                x=h,                                   # Current predictor state
                decoder_output=decoder_output_causal,  # K/V source for cross-attention (causal backbone output)
                self_attention_mask=attention_mask,    # Padding mask for predictor's CAUSAL self-attention
                cross_attention_mask=context_mask      # JEPA context mask to select K/V in cross-attention
            )

        # Apply final layer normalization to the predictor's output
        h = self.ln_f(h) # [B, T, C]

        # --- 3. Extract Predicted Embeddings for Target Spans ---
        predicted_spans_embeddings: List[List[torch.Tensor]] = []
        for b in range(B): # Iterate through batch items
            batch_spans_preds = []
            # Check if there are any target spans defined for this batch item
            if not target_spans_indices[b]:
                predicted_spans_embeddings.append(batch_spans_preds) # Append empty list
                continue

            # Extract embeddings corresponding to each target span
            for start, end in target_spans_indices[b]:
                # Ensure indices are valid and span has non-zero length
                valid_end = min(end, T) # Clamp end index to sequence length
                if start < valid_end:
                    # Extract the predicted embeddings from the predictor's output `h`
                    span_pred_emb = h[b, start:valid_end] # Shape [SpanLen, C]
                    batch_spans_preds.append(span_pred_emb)

            predicted_spans_embeddings.append(batch_spans_preds)

        return predicted_spans_embeddings

# ==========================================
# 8) Target Encoder (EMA copy of BackboneDecoder, runs NON-CAUSALLY)
# ==========================================
class TargetEncoder(nn.Module):
    """
    Target Encoder for JEPA.

    This module maintains an Exponential Moving Average (EMA) copy of the
    BackboneDecoder's weights. It is used to generate the target embeddings
    for the JEPA prediction task. Crucially, it runs the underlying decoder
    in a NON-CAUSAL (bidirectional) manner to produce contextually rich target
    representations. Its parameters are not updated by the optimizer directly;
    they are updated via the `update_ema` method.

    Args:
        backbone_decoder (BackboneDecoder): An instance of the BackboneDecoder
                                            whose weights will be tracked via EMA.
        ema_decay (float): The decay rate for the EMA update. A higher value means
                           slower updates. Default is 0.999.
    """
    def __init__(self, backbone_decoder: BackboneDecoder, ema_decay: float = 0.999):
        super().__init__()
        # Create a deep copy of the backbone decoder's structure and initial weights
        self.encoder = copy.deepcopy(backbone_decoder)
        self.ema_decay = ema_decay

        # Disable gradient computation for all parameters in the target encoder
        # This ensures it's not trained directly by the optimizer.
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"Target Encoder (EMA) initialized with decay {ema_decay}.")

    @torch.no_grad() # Ensure no gradients are computed during EMA update
    def update_ema(self, backbone_decoder: BackboneDecoder, decay_rate: Optional[float] = None):
        """
        Updates the target encoder's weights using EMA with the backbone decoder's weights.

        Formula: target_param = decay * target_param + (1 - decay) * source_param

        Args:
            backbone_decoder (BackboneDecoder): The source model (online/trained backbone)
                                               from which to update the weights.
            decay_rate (Optional[float]): The EMA decay rate to use for this update.
                                          If None, uses the instance's default `ema_decay`.
                                          Defaults to None.
        """
        decay = decay_rate if decay_rate is not None else self.ema_decay

        # Ensure both models are in eval mode for consistent state (e.g., dropout disabled)
        # although target encoder params don't have grads anyway.
        self.encoder.eval()
        backbone_decoder.eval()

        # Get named parameters from both models
        source_params = dict(backbone_decoder.named_parameters())
        target_params = dict(self.encoder.named_parameters())

        # Check if parameter names match (important for correct update)
        assert source_params.keys() == target_params.keys(), \
            "Parameter names mismatch between backbone and target encoders!"

        # Apply EMA update rule to each parameter
        for name, source_param in source_params.items():
            # Ensure the parameter exists in the target encoder
            if name in target_params:
                target_param = target_params[name]
                # Update in-place: target = decay * target + (1 - decay) * source
                target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)
            else:
                 # This should not happen due to the assert above, but as a safeguard:
                 print(f"Warning: Parameter '{name}' not found in target encoder during EMA update.")

        # It's good practice to switch the backbone back to train mode if needed outside this function.


    @torch.no_grad() # Target encoder forward pass should not compute gradients
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a NON-CAUSAL forward pass using the EMA weights.

        This generates the target representations for the JEPA loss.

        Args:
            x (torch.Tensor): Input token indices. Shape [B, T].
            attention_mask (Optional[torch.Tensor]): Padding mask. Shape [B, T].
                                                     1 for real tokens, 0 for padding.
                                                     Defaults to None.

        Returns:
            torch.Tensor: Output embeddings computed non-causally. Shape [B, T, C].
        """
        # Ensure the target encoder is always in evaluation mode
        self.encoder.eval()
        # Call the underlying decoder's forward pass, explicitly setting is_causal=False
        # to get bidirectional representations.
        return self.encoder(x, attention_mask=attention_mask, is_causal=False)

# ==========================================
# 9) Complete T-JEPA Model (Decoder Backbone)
# ==========================================
class TJEPAModel(nn.Module):
    """
    The complete Transformer JEPA (T-JEPA) model with a Decoder backbone.

    Integrates the following components:
    - BackboneDecoder: The main trainable decoder model (runs causally for LM/predictor).
    - JEPAPredictor: Predicts target span representations from context.
    - TargetEncoder: EMA copy of the BackboneDecoder (runs non-causally for JEPA targets).
    - LM Head: Predicts the next token based on the BackboneDecoder's output.

    Performs multi-task learning:
    - JEPA Loss: MSE loss between predicted embeddings (from Predictor) and
                 target embeddings (from TargetEncoder).
    - LM Loss: Cross-entropy loss for next-token prediction.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of embeddings and hidden states.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of layers in the backbone and predictor.
        block_size (int): Maximum sequence length (context window).
        ema_decay (float): Decay rate for the TargetEncoder's EMA update. Default is 0.999.
        lm_loss_weight (float): Weighting factor for the LM loss component in the
                                total loss. Default is 0.1.
        pad_token_id (int): The ID of the padding token, used to ignore padding
                            in the LM loss calculation. Default is 0.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 n_heads: int,
                 n_layers: int,
                 block_size: int,
                 ema_decay: float = 0.999,
                 lm_loss_weight: float = 0.1,
                 pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.lm_loss_weight = lm_loss_weight
        self.block_size = block_size # Store block_size for generation checks

        # --- Instantiate Components ---
        # 1. Main Backbone: Transformer Decoder (trainable)
        self.decoder_backbone = BackboneDecoder(vocab_size, embed_dim, n_heads, n_layers, block_size)

        # 2. JEPA Predictor (trainable)
        self.predictor = JEPAPredictor(embed_dim, n_heads, n_layers, block_size)

        # 3. Target Encoder (EMA copy of backbone, not directly trainable)
        self.target_encoder = TargetEncoder(self.decoder_backbone, ema_decay)
        # Perform an initial weight copy (decay=0) to synchronize target encoder at creation
        self.target_encoder.update_ema(self.decoder_backbone, decay_rate=0.0)
        print("Initial Target Encoder weights synchronized with Backbone.")

        # 4. Language Modeling Head (trainable)
        # Projects final decoder embeddings to vocabulary logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # --- Weight Tying ---
        # Tie the weights of the token embedding layer and the LM head
        # This improves performance and reduces parameter count.
        self.decoder_backbone.token_embedding.weight = self.lm_head.weight
        print("Weights tied between BackboneDecoder token embedding and LM Head.")

    def forward(self,
                x: torch.Tensor,
                context_mask: torch.Tensor,
                target_spans_indices: List[List[Tuple[int, int]]],
                attention_mask: torch.Tensor) -> Dict[str, Any]:
        """
        Performs the forward pass for training, calculating outputs needed for both
        JEPA and LM losses.

        Args:
            x (torch.Tensor): Input token sequence. Shape [B, T].
            context_mask (torch.Tensor): JEPA context mask. Shape [B, T].
                                         (1=context, 0=target/pad).
            target_spans_indices (List[List[Tuple[int, int]]]): List of target span indices.
            attention_mask (torch.Tensor): Padding mask. Shape [B, T]. (1=real, 0=pad).

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "predicted_spans_embeddings": List[List[Tensor]] from Predictor.
                - "target_spans_embeddings": List[List[Tensor]] from Target Encoder (non-causal).
                - "lm_logits": Tensor [B, T, VocabSize] from Backbone Decoder (causal) + LM Head.
                - "input_sequence": The input tensor `x` [B, T], needed for LM loss.
                - "attention_mask": The padding mask [B, T], potentially useful for loss calc.
        """

        # --- 1. Causal Backbone Pass ---
        # Run the main decoder backbone causally.
        # Output used for: a) LM loss, b) Predictor's cross-attention context.
        decoder_output_causal = self.decoder_backbone(
            x,
            attention_mask=attention_mask,
            is_causal=True # Standard causal operation for LM and predictor context
        ) # Shape: [B, T, C]

        # --- 2. Non-Causal Target Encoder Pass ---
        # Run the EMA target encoder non-causally (no gradients).
        # Output used for: a) JEPA target embeddings.
        with torch.no_grad(): # Ensure no gradients are computed for target encoder
            self.target_encoder.eval() # Set target encoder to evaluation mode
            target_embeddings_full = self.target_encoder(
                x,
                attention_mask=attention_mask
                # The target_encoder's forward method internally calls its backbone with is_causal=False
            ) # Shape: [B, T, C]

        # --- 3. Predictor Pass ---
        # Run the JEPA predictor.
        # Predicts target span embeddings using the causal backbone output as context.
        predicted_spans_embeddings = self.predictor(
            decoder_output_causal=decoder_output_causal, # Context for cross-attention
            target_spans_indices=target_spans_indices,   # Specifies which spans to predict
            context_mask=context_mask,                   # Mask for cross-attention K/V selection
            attention_mask=attention_mask                # Padding mask for predictor's self-attention
        ) # Output: List[List[Tensor]]

        # --- 4. Extract Actual Target Embeddings ---
        # Extract the embeddings for the target spans from the *non-causal* target encoder output.
        target_spans_embeddings: List[List[torch.Tensor]] = []
        B = x.size(0)
        T = x.size(1)
        for b in range(B): # Iterate through batch items
            batch_target_spans = []
            if not target_spans_indices[b]: # Handle cases with no spans for this item
                target_spans_embeddings.append(batch_target_spans)
                continue
            # Extract target embeddings corresponding to each span
            for start, end in target_spans_indices[b]:
                valid_end = min(end, T) # Ensure end index is within sequence bounds
                if start < valid_end: # Ensure span has non-zero length
                    # Extract from the full target embeddings (non-causal pass result)
                    span_target_emb = target_embeddings_full[b, start:valid_end] # Shape: [SpanLen, C]
                    batch_target_spans.append(span_target_emb)
            target_spans_embeddings.append(batch_target_spans)

        # --- 5. Calculate LM Logits ---
        # Project the output of the *causal* backbone pass through the LM head.
        lm_logits = self.lm_head(decoder_output_causal) # Shape: [B, T, VocabSize]

        # --- Return all necessary outputs for loss calculation ---
        return {
            "predicted_spans_embeddings": predicted_spans_embeddings, # From Predictor [List[List[Tensor[SpanLen, C]]]]
            "target_spans_embeddings": target_spans_embeddings,     # From Target Encoder (non-causal) [List[List[Tensor[SpanLen, C]]]]
            "lm_logits": lm_logits,                                 # From Backbone (causal) + LM Head [B, T, V]
            "input_sequence": x,                                    # Original input tokens [B, T]
            "attention_mask": attention_mask,                       # Padding mask [B, T]
        }

    def update_target_encoder(self):
        """Convenience method to update the target encoder's EMA weights."""
        self.target_encoder.update_ema(self.decoder_backbone)

    def compute_loss(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes the combined JEPA (MSE) and Language Modeling (CrossEntropy) loss.

        Args:
            outputs (Dict[str, Any]): The dictionary returned by the `forward` method.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "total_loss": The combined weighted loss.
                - "jepa_loss": The Mean Squared Error loss for JEPA.
                - "lm_loss": The Cross Entropy loss for Language Modeling.
        """
        # --- 1. JEPA MSE Loss ---
        predicted_spans = outputs["predicted_spans_embeddings"] # List[List[Tensor[SpanLen, C]]]
        target_spans = outputs["target_spans_embeddings"]       # List[List[Tensor[SpanLen, C]]]
        batch_size = len(predicted_spans)
        jepa_losses_per_item = [] # Store avg loss for each batch item with valid spans
        num_valid_span_comparisons = 0 # Count how many span pairs actually contribute to loss

        for b in range(batch_size):
            num_spans_in_item = len(predicted_spans[b])
            # Basic sanity check: should have same number of predicted and target spans
            if num_spans_in_item != len(target_spans[b]):
                # This might happen if span generation/extraction had issues. Log or handle as needed.
                # print(f"Warning: Mismatch in num predicted ({num_spans_in_item}) vs target ({len(target_spans[b])}) spans for batch item {b}. Skipping JEPA loss for this item.")
                continue # Skip JEPA loss calculation for this batch item

            if num_spans_in_item == 0:
                continue # Skip if no spans were generated/extracted for this item

            span_losses_for_item = [] # Store MSE loss for each valid span pair in this item
            for i in range(num_spans_in_item):
                pred_span_emb = predicted_spans[b][i] # [SpanLen_pred, C]
                target_span_emb = target_spans[b][i]  # [SpanLen_target, C]

                # Ensure spans are not empty and shapes match exactly before computing MSE
                # Shape mismatch could occur if span boundaries differ slightly due to edge cases
                if pred_span_emb.nelement() > 0 and target_span_emb.nelement() > 0 and pred_span_emb.shape == target_span_emb.shape:

                    # Optional: Normalize embeddings before MSE loss (cosine similarity loss)
                    # pred_span_norm = F.normalize(pred_span_emb, p=2, dim=-1)
                    # target_span_norm = F.normalize(target_span_emb, p=2, dim=-1)
                    # loss = F.mse_loss(pred_span_norm, target_span_norm) # (1 - cosine_sim) is another option

                    # Calculate standard MSE loss between predicted and target embeddings
                    loss = F.mse_loss(pred_span_emb, target_span_emb)
                    span_losses_for_item.append(loss)
                    num_valid_span_comparisons += 1
                # else:
                    # Log details if a span comparison is skipped
                    # if pred_span_emb.nelement() == 0 or target_span_emb.nelement() == 0:
                    #      print(f"Debug: Skipped empty span comparison at batch {b}, span {i}.")
                    # elif pred_span_emb.shape != target_span_emb.shape:
                    #      print(f"Debug: Skipped shape mismatch {pred_span_emb.shape} vs {target_span_emb.shape} at batch {b}, span {i}.")
                    # else:
                    #      print(f"Debug: Skipped span comparison at batch {b}, span {i} for unknown reason.")


            # Average loss across valid spans *for this specific batch item*
            if span_losses_for_item:
                 jepa_losses_per_item.append(torch.stack(span_losses_for_item).mean())

        # Average JEPA loss over the batch items that had at least one valid span comparison
        if jepa_losses_per_item:
            final_jepa_loss = torch.stack(jepa_losses_per_item).mean()
        else:
            # If NO valid span comparisons occurred in the entire batch, JEPA loss is 0.
            # Return a tensor with requires_grad=False if needed, or just 0.0.
            example_tensor = outputs["lm_logits"] # Get device/dtype from another tensor
            final_jepa_loss = torch.tensor(0.0, device=example_tensor.device, dtype=example_tensor.dtype, requires_grad=False)
            # if num_valid_span_comparisons == 0:
            #      print("Warning: JEPA loss is 0.0 because no valid span comparisons occurred in this batch.")

        # --- 2. Language Modeling (LM) Cross Entropy Loss ---
        lm_logits = outputs["lm_logits"]           # [B, T, VocabSize]
        input_sequence = outputs["input_sequence"] # [B, T]

        # Reshape for CrossEntropyLoss:
        # Predict token at position `t` based on output at `t-1`.
        # Shift logits left (remove last prediction) and labels left (remove first token).
        # Logits: [B, T, V] -> [B, T-1, V]
        # Labels: [B, T]    -> [B, T-1]
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = input_sequence[:, 1:].contiguous()

        # Flatten the batch and sequence dimensions for the loss function
        # Logits: [B, T-1, V] -> [B * (T-1), V]
        # Labels: [B, T-1]    -> [B * (T-1)]
        vocab_size = shift_logits.size(-1)
        shift_logits_flat = shift_logits.view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)

        # Calculate Cross Entropy loss, ignoring padding tokens in the labels
        lm_loss = F.cross_entropy(shift_logits_flat, shift_labels_flat, ignore_index=self.pad_token_id)

        # --- 3. Combine Losses ---
        # Weighted sum of JEPA loss and LM loss
        # Ensure final_jepa_loss requires grad if it should contribute (it will if derived from trainable params)
        # If final_jepa_loss is tensor(0.0), adding it won't hurt.
        total_loss = final_jepa_loss * (1.0 - self.lm_loss_weight) + lm_loss * self.lm_loss_weight
        # Alternative weighting: total_loss = jepa_loss + lm_weight * lm_loss
        # total_loss = final_jepa_loss + self.lm_loss_weight * lm_loss


        return {
            "total_loss": total_loss,
            "jepa_loss": final_jepa_loss.detach(), # Detach for logging purposes if needed
            "lm_loss": lm_loss.detach()            # Detach for logging purposes
        }

    @torch.no_grad() # Generation should not compute gradients
    def generate(self,
                 x: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generates text autoregressively starting from a given context sequence `x`.

        Uses the BackboneDecoder and LM Head for generation. Employs temperature
        scaling and top-p (nucleus) sampling.

        Args:
            x (torch.Tensor): The initial context sequence (prompt) as token IDs.
                              Shape [B, T_prompt], where B is batch size.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): Sampling temperature. Lower values make the output
                                 more deterministic, higher values make it more random.
                                 Setting to 0 disables randomness (becomes greedy).
                                 Default is 1.0 (no change).
            top_p (Optional[float]): Nucleus sampling threshold. If set (e.g., 0.9), only
                                     tokens comprising the top `top_p` probability mass
                                     are considered for sampling. Set to None or 1.0 to
                                     disable. Default is None.

        Returns:
            torch.Tensor: The generated sequence, including the initial context `x`.
                          Shape [B, T_prompt + T_generated].
        """
        self.eval() # Ensure the model is in evaluation mode (disables dropout, etc.)
        B = x.size(0) # Batch size
        pad_token_id = self.pad_token_id # Get pad token ID from instance

        generated_sequence = x # Start with the initial context

        for _ in range(max_new_tokens):
            # --- Prepare Input for this Step ---
            # Crop the context if it exceeds the model's block size.
            # Take the last `block_size` tokens as context for the next prediction.
            current_context = generated_sequence if generated_sequence.size(1) <= self.block_size else generated_sequence[:, -self.block_size:]
            seq_len = current_context.size(1)

            # Create attention mask for the current context (masking padding tokens)
            # Shape [B, seq_len], 1 for real tokens, 0 for padding
            attention_mask = (current_context != pad_token_id).float()

            # --- Forward Pass (Causal) ---
            # Get embeddings from the decoder backbone, ensuring causal masking
            decoder_output = self.decoder_backbone(
                current_context,
                attention_mask=attention_mask,
                is_causal=True # Must be causal for autoregressive generation
            ) # Shape: [B, seq_len, C]

            # --- Get Logits for the Next Token ---
            # Use the embedding of the *last* token in the current context sequence
            # to predict the *next* token.
            # decoder_output[:, -1, :] gives shape [B, C]
            # lm_head projects [B, C] -> [B, VocabSize]
            logits = self.lm_head(decoder_output[:, -1, :])

            # --- Apply Temperature Scaling ---
            # Divide logits by temperature. Lower temp -> sharper distribution.
            if temperature > 0 and temperature != 1.0:
                 logits = logits / temperature
            # Handle temperature = 0 (greedy decoding) - becomes argmax after softmax
            elif temperature == 0:
                 # Taking argmax directly is equivalent to temp=0 after softmax
                 next_token = torch.argmax(logits, dim=-1, keepdim=True) # [B, 1]
                 generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
                 continue # Skip sampling if greedy

            # --- Apply Top-p (Nucleus) Sampling ---
            if top_p is not None and 0.0 < top_p < 1.0:
                # 1. Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                # 2. Calculate cumulative probabilities of sorted logits
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 3. Create a mask for tokens to remove (those falling outside the top-p nucleus)
                # Find tokens where cumulative probability exceeds top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the mask right: always keep the token that *just* crossed the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0 # Never remove the highest probability token

                # 4. Scatter the mask back to the original vocabulary order
                # Create a boolean mask of the same shape as logits
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )

                # 5. Apply the mask by setting logits of low-probability tokens to -infinity
                logits[indices_to_remove] = float('-inf')

            # --- Sample the Next Token ---
            # Convert final logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample one token for each sequence in the batch based on the probabilities
            next_token = torch.multinomial(probs, num_samples=1) # Shape: [B, 1]

            # --- Append Sampled Token and Continue ---
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

            # --- Optional: Early Stopping Check ---
            # Check if the End-of-Sequence token was generated in *all* batch items
            # if hyperparams.get('eos_token') is not None and (next_token == hyperparams['eos_token']).all():
            #     break # Stop generation for the whole batch if all sequences finished

        return generated_sequence


# ==========================================
# 10) Evaluation Function
# ==========================================
@torch.no_grad() # Ensure no gradients are computed during evaluation
def estimate_loss(model: TJEPAModel, train_df: pd.DataFrame, val_df: pd.DataFrame, hyperparams: Dict[str, Any], device: str) -> Dict[str, float]:
    """
    Estimates the average loss (total, JEPA, LM) on the training and validation sets.

    Runs the model in evaluation mode over a fixed number of iterations (`eval_iters`)
    for each data split.

    Args:
        model (TJEPAModel): The model to evaluate.
        train_df (pd.DataFrame): The training dataset DataFrame.
        val_df (pd.DataFrame): The validation dataset DataFrame.
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters, including
                                     `eval_iters`, `batch_size`, etc.
        device (str): The compute device.

    Returns:
        Dict[str, float]: A dictionary containing the average losses for each split,
                          e.g., {'train_total': 1.23, 'train_jepa': 0.45, ... ,
                          'val_total': 1.30, 'val_jepa': 0.50, ...}.
    """
    out = {}
    model.eval() # Set the model to evaluation mode (disables dropout, etc.)

    eval_iters = hyperparams['eval_iters']
    print(f"Running evaluation for {eval_iters} iterations...")

    for split, df in [('train', train_df), ('val', val_df)]:
        # Initialize tensors to store losses for each iteration
        total_losses = torch.zeros(eval_iters, device=device)
        jepa_losses = torch.zeros(eval_iters, device=device)
        lm_losses = torch.zeros(eval_iters, device=device)

        # Loop for the specified number of evaluation iterations
        pbar_eval = tqdm(range(eval_iters), desc=f"Eval {split}", leave=False, ncols=80)
        for k in pbar_eval:
            try:
                # Get a batch of data from the current split's DataFrame
                x, context_mask, target_spans_indices, attention_mask = prepare_batches_from_gsm8k(
                    df, hyperparams, device
                )

                # Perform a forward pass through the model
                outputs = model(x, context_mask, target_spans_indices, attention_mask)

                # Compute the loss using the model's internal loss function
                loss_dict = model.compute_loss(outputs)

                # Store the computed losses for this batch
                total_losses[k] = loss_dict['total_loss'] # Use detached loss from compute_loss
                jepa_losses[k] = loss_dict['jepa_loss']
                lm_losses[k] = loss_dict['lm_loss']

            except Exception as e:
                print(f"\nError during evaluation step {k} for split {split}: {e}")
                # Store NaN or skip this iteration's result
                total_losses[k] = float('nan')
                jepa_losses[k] = float('nan')
                lm_losses[k] = float('nan')


        # Calculate the mean loss over all evaluation iterations, ignoring NaNs
        out[f'{split}_total'] = torch.nanmean(total_losses).item()
        out[f'{split}_jepa'] = torch.nanmean(jepa_losses).item()
        out[f'{split}_lm'] = torch.nanmean(lm_losses).item()

    # It's the responsibility of the caller to set the model back to train mode if needed
    # model.train()
    return out


# ==========================================
# 11) Generate Text Function (Uses model.generate)
# ==========================================
@torch.no_grad() # Ensure no gradients are computed during generation
def generate_from_prompt(model: TJEPAModel,
                         hyperparams: Dict[str, Any],
                         prompt_text: Optional[str] = None,
                         max_new_tokens: Optional[int] = None,
                         top_p: Optional[float] = None,
                         temperature: Optional[float] = None,
                         device: str = "cuda") -> str:
    """
    Generates text starting from a given prompt using the model's `generate` method.

    Handles encoding the prompt, calling `model.generate`, and decoding the result.

    Args:
        model (TJEPAModel): The trained model instance.
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters. Used for default
                                     prompt, generation settings (top_p, temp, max_tokens),
                                     and special tokens (BOS, EOS, PAD).
        prompt_text (Optional[str]): The text prompt to start generation from. If None,
                                     uses `hyperparams['start_prompt']`. Defaults to None.
        max_new_tokens (Optional[int]): Maximum number of new tokens to generate. If None,
                                       uses `hyperparams['generate_num_tokens']`. Defaults to None.
        top_p (Optional[float]): Top-p sampling threshold. If None, uses `hyperparams['top_p']`.
                                Defaults to None.
        temperature (Optional[float]): Sampling temperature. If None, uses
                                      `hyperparams['temperature']`. Defaults to None.
        device (str): The compute device. Defaults to "cuda".

    Returns:
        str: The generated text, including the prompt, decoded from bytes.
             Returns a byte string representation if decoding fails.
    """
    model.eval() # Ensure the model is in evaluation mode

    # --- Determine Generation Parameters ---
    prompt = prompt_text if prompt_text is not None else hyperparams['start_prompt']
    max_tokens = max_new_tokens if max_new_tokens is not None else hyperparams['generate_num_tokens']
    p_val = top_p if top_p is not None else hyperparams['top_p']
    temp_val = temperature if temperature is not None else hyperparams['temperature']
    system_prompt = hyperparams['system_prompt']
    think_tag = hyperparams['thinking_tag']

    # --- Prepare Prompt ---
    # Prepend system prompt and format for GSM8K structure, ending before the reasoning starts
    full_prompt = f"{system_prompt}\n\nProblem: {prompt}\n\n{think_tag}" # Start generation inside think tags

    # --- Encode Prompt ---
    bos_token = hyperparams['bos_token']
    # Encode the formatted prompt string to bytes, prepend BOS token
    prompt_bytes = [bos_token] + list(full_prompt.encode('utf-8', errors='replace'))
    # Convert byte list to a tensor, add batch dimension [1, T_prompt]
    context = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0)

    # --- Generate Tokens ---
    print(f"Generating up to {max_tokens} tokens with top_p={p_val}, temp={temp_val}...")
    # Call the model's built-in generate method
    full_output_tokens_tensor = model.generate(
        context,
        max_new_tokens=max_tokens,
        top_p=p_val,
        temperature=temp_val
    ) # Output shape: [1, T_prompt + T_generated]

    # --- Decode Output ---
    # Get the generated token list (including prompt) from the tensor
    full_output_list = full_output_tokens_tensor[0].tolist()

    # Decode the byte sequence back to a UTF-8 string
    try:
        # Find EOS token if present and truncate the sequence there
        eos_token = hyperparams['eos_token']
        eos_pos = -1
        if eos_token in full_output_list:
            eos_pos = full_output_list.index(eos_token)

        if eos_pos != -1:
            # Truncate the list at the first occurrence of EOS
            full_output_list = full_output_list[:eos_pos]

        # Filter out padding tokens (though usually not generated unless EOS hit early)
        # and remove the initial BOS token for cleaner output presentation.
        pad_token = hyperparams['pad_token']
        decoded_bytes = bytes([tok for tok in full_output_list if tok != pad_token and tok != bos_token]) # Remove BOS & PAD

        # Attempt to decode the byte sequence using UTF-8, replacing errors
        generated_text = decoded_bytes.decode('utf-8', errors='replace')
        return generated_text

    except Exception as e:
        print(f"Error during decoding of generated sequence: {e}")
        # Fallback: return the raw byte string representation if decoding fails
        return str(bytes(full_output_list))

# ==========================================
# 12) Token-by-Token Generation (Manual loop for streaming)
# ==========================================
@torch.no_grad() # Ensure no gradients are computed during generation
def generate_token_by_token(model: TJEPAModel,
                            hyperparams: Dict[str, Any],
                            prompt_text: str,
                            max_new_tokens: int = 200,
                            device: str = "cuda") -> str:
    """
    Generates text token by token (byte by byte) using a manual loop,
    printing the output progressively (simulating streaming).

    This provides a visual demonstration of the autoregressive generation process.
    It manually implements the sampling logic (temperature, top-p) within the loop.

    Args:
        model (TJEPAModel): The trained model instance.
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters for generation
                                     settings and special tokens.
        prompt_text (str): The text prompt to start generation from.
        max_new_tokens (int): Maximum number of new tokens (bytes) to generate.
                              Defaults to 200.
        device (str): The compute device. Defaults to "cuda".

    Returns:
        str: The complete generated text (including the prompt) after the loop finishes
             or EOS is reached. Returns byte string representation if final decoding fails.
    """
    model.eval() # Ensure the model is in evaluation mode

    # --- Get Generation Parameters ---
    system_prompt = hyperparams['system_prompt']
    think_tag = hyperparams['thinking_tag']
    bos_token = hyperparams['bos_token']
    pad_token = hyperparams['pad_token']
    eos_token = hyperparams['eos_token']
    top_p = hyperparams['top_p']
    temperature = hyperparams['temperature']
    block_size = model.block_size # Get block size from the model instance

    # --- Prepare and Encode Prompt ---
    full_prompt = f"{system_prompt}\n\nProblem: {prompt_text}\n\n{think_tag}"
    prompt_bytes = [bos_token] + list(full_prompt.encode('utf-8', errors='replace'))
    context = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0) # Shape: [1, T_prompt]

    print(f"\n--- Generating token-by-token (max {max_new_tokens}) ---")
    print(full_prompt, end="", flush=True) # Print the initial prompt without newline

    generated_tokens = [] # Keep track of generated token IDs
    current_byte_fragment = b'' # Accumulator for multi-byte UTF-8 characters

    # --- Manual Generation Loop ---
    for i in range(max_new_tokens):
        # --- Prepare Input for this Step ---
        # Crop context if it exceeds block size
        context_cond = context if context.size(1) <= block_size else context[:, -block_size:]
        # Create attention mask for padding
        attention_mask = (context_cond != pad_token).float() # Shape [1, T_cond]

        # --- Forward Pass (Causal) ---
        decoder_output = model.decoder_backbone(
            context_cond,
            attention_mask=attention_mask,
            is_causal=True # Causal is essential for generation
        ) # Shape: [1, T_cond, C]
        # Get logits for the *next* token using the *last* token's output embedding
        logits = model.lm_head(decoder_output[:, -1, :]) # Shape: [1, VocabSize]

        # --- Sampling Logic (Temperature and Top-p) ---
        if temperature == 0: # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            # Apply Top-p
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from the modified distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # Shape: [1, 1]

        # --- Update Context and Decode/Print ---
        next_token_value = next_token.item() # Get the integer token ID
        # Append the new token ID to the context for the next iteration
        context = torch.cat([context, next_token], dim=1)
        generated_tokens.append(next_token_value)

        # Attempt to decode and print the new byte(s) immediately
        current_byte_fragment += bytes([next_token_value])
        try:
            # Try decoding the accumulated byte fragment
            next_char = current_byte_fragment.decode('utf-8')
            # If successful, print the character(s) and reset the fragment
            print(next_char, end="", flush=True)
            current_byte_fragment = b'' # Reset fragment
            time.sleep(0.01) # Small delay for better visualization
        except UnicodeDecodeError:
            # If decoding fails, it's likely a partial multi-byte character.
            # Continue accumulating bytes in the next iteration.
            # Handle potential issues with long invalid sequences:
            if len(current_byte_fragment) > 4: # Max UTF-8 char length is 4 bytes
                 print("<?>", end="", flush=True) # Print placeholder for likely invalid sequence
                 current_byte_fragment = b'' # Reset fragment to avoid getting stuck

        # --- Check for End-of-Sequence Token ---
        if next_token_value == eos_token:
            print(" <EOS>", end="", flush=True)
            break # Stop generation if EOS token is produced

    print("\n--- Token-by-token generation completed ---")

    # --- Final Decoding of the Full Sequence ---
    # Combine prompt bytes and generated token bytes
    full_generated_list = prompt_bytes + generated_tokens
    try:
        # Find EOS token and truncate if present
        eos_pos = -1
        if eos_token in full_generated_list:
           eos_pos = full_generated_list.index(eos_token)
        if eos_pos != -1:
            full_generated_list = full_generated_list[:eos_pos]

        # Remove padding and BOS tokens, then decode the complete sequence
        pad_token = hyperparams['pad_token']
        bos_token = hyperparams['bos_token']
        # Keep generated tokens, remove padding and the initial BOS
        decoded_bytes = bytes([tok for tok in full_generated_list if tok != pad_token]) # Keep BOS maybe?

        # Decode the final byte sequence
        final_text = decoded_bytes.decode('utf-8', errors='replace')
        return final_text
    except Exception as e:
        print(f"Final decoding error after token-by-token generation: {e}")
        # Fallback to byte string representation
        return str(bytes(full_generated_list))


# ==========================================
# 13) Training Implementation
# ==========================================
def train(continue_training: bool = True):
    """
    Trains the T-JEPA DECODER model on the GSM8K dataset.

    Handles model initialization, optimizer setup, checkpoint loading,
    the main training loop (epochs and steps), evaluation, learning rate scheduling,
    gradient accumulation, gradient clipping, EMA updates, and checkpoint saving.

    Args:
        continue_training (bool): If True and a checkpoint exists at the path defined
                                  in hyperparameters, training will resume from there.
                                  Otherwise, starts training from scratch. Defaults to True.
    """
    # --- Setup ---
    print("Initializing training setup...")
    hyperparams = get_hyperparams()
    device = get_device()
    train_df, val_df, test_df = load_gsm8k_data()

    # --- Model Initialization ---
    print("Initializing T-JEPA model...")
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

    # Print model size information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Configuration:")
    print(f"  Block Size: {hyperparams['block_size']}")
    print(f"  Embedding Dim: {hyperparams['embed_dim']}")
    print(f"  Layers: {hyperparams['n_layers']}, Heads: {hyperparams['n_heads']}")
    print(f"  Vocab Size: {hyperparams['vocab_size']}")
    print(f"Total parameters: {total_params:,}")
    # Trainable params exclude the Target Encoder
    print(f"Trainable parameters (Backbone + Predictor + LM Head): {trainable_params:,}")

    # --- Optimizer ---
    # AdamW is commonly used for Transformers
    # Filter parameters to only include those requiring gradients (excludes TargetEncoder)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hyperparams['learning_rate'], # Initial LR, scheduler will adjust
        betas=(0.9, 0.95),              # Recommended betas for AdamW
        weight_decay=hyperparams['weight_decay'] # Weight decay (L2 regularization)
    )
    print("Optimizer AdamW initialized.")

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_loss = float('inf')
    current_step = 0 # Global step counter across epochs
    checkpoint_path = hyperparams['checkpoint_path']

    if continue_training and os.path.exists(checkpoint_path):
        print(f"Attempting to load checkpoint from {checkpoint_path}...")
        try:
            # Load checkpoint onto the correct device
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_state = checkpoint['model_state']

            # --- Flexible State Dict Loading ---
            # Allows loading even if some keys changed (e.g., layer names) or are missing/extra.
            print("Loading model state dict (flexibly)...")
            current_model_dict = model.state_dict()
            processed_state_dict = {}
            warned_keys_shape = set()
            warned_keys_missing = set()

            for k, v in model_state.items():
                new_k = k
                # Example adaptation: If checkpoint used 'context_encoder' instead of 'decoder_backbone'
                if k.startswith("context_encoder."):
                    new_k = k.replace("context_encoder.", "decoder_backbone.", 1)
                    # print(f"  Mapping checkpoint key '{k}' to '{new_k}'") # Optional debug log

                if new_k in current_model_dict:
                    # Check if shapes match before assigning
                    if v.shape == current_model_dict[new_k].shape:
                        processed_state_dict[new_k] = v
                    else:
                        # Log shape mismatch warning only once per key
                        if new_k not in warned_keys_shape:
                            print(f"  Warning: Shape mismatch for key '{new_k}'. "
                                  f"Checkpoint shape: {v.shape}, Model shape: {current_model_dict[new_k].shape}. "
                                  f"Skipping this parameter.")
                            warned_keys_shape.add(new_k)
                else:
                    # Log missing key warning only once per key
                    original_key_str = f" (original: '{k}')" if new_k != k else ""
                    if new_k not in warned_keys_missing:
                         print(f"  Warning: Key '{new_k}'{original_key_str} from checkpoint not found in current model. Skipping.")
                         warned_keys_missing.add(new_k)
                         if k != new_k: warned_keys_missing.add(k) # Avoid double warning if rename failed

            # Load the processed state dict (non-strict mode)
            missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)

            # Report keys that were in the model but not loaded from the processed checkpoint
            if missing_keys:
                print(f"  Warning: The following keys were missing in the loaded checkpoint state_dict "
                      f"(using initialized values): {missing_keys}")
            # Report keys that were loaded but are not expected by the current model (should be empty with the filtering logic)
            if unexpected_keys:
                print(f"  Warning: The following keys from the checkpoint were unexpected by the model: {unexpected_keys}")
            print("Model state loaded.")
            # --- End Flexible State Dict Loading ---

            # Load optimizer state cautiously (only if structure seems compatible)
            print("Attempting to load optimizer state...")
            try:
                if 'optimizer_state' in checkpoint:
                    # Basic check: does the number of parameter groups match?
                    if len(optimizer.param_groups) == len(checkpoint['optimizer_state']['param_groups']):
                        # Add more checks here if needed (e.g., compare shapes of state tensors)
                        optimizer.load_state_dict(checkpoint['optimizer_state'])
                        print("  Optimizer state loaded successfully.")
                    else:
                         print("  Warning: Optimizer parameter group count mismatch. Reinitializing optimizer state.")
                         # Keep the initialized optimizer
                else:
                    print("  Warning: Optimizer state not found in checkpoint. Using initialized optimizer.")
            except Exception as e_optim:
                 print(f"  Warning: Could not load optimizer state due to error: {e_optim}. Using initialized optimizer.")
                 # Keep the initialized optimizer

            # Load training progress
            start_epoch = checkpoint.get('epoch', 0) + 1 # Resume from the next epoch
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            # Use current_step if saved, otherwise estimate from epoch
            current_step = checkpoint.get('current_step', start_epoch * hyperparams['steps_per_epoch'])
            loaded_hyperparams = checkpoint.get('hyperparams', {})
            # You might want to compare loaded_hyperparams with current ones or selectively use them.
            # For now, we primarily use the current hyperparams but load the training state.
            print(f"Resuming training from Epoch {start_epoch}, Global Step {current_step}.")
            print(f"  Best validation loss recorded: {best_val_loss:.4f}")

            # --- IMPORTANT: Re-sync target encoder ---
            # After loading the trained backbone weights, ensure the target encoder
            # reflects these loaded weights, not the initial random weights or EMA state
            # from before loading. Decay=0.0 forces a direct copy.
            print("Re-synchronizing Target Encoder with loaded Backbone weights...")
            model.target_encoder.update_ema(model.decoder_backbone, decay_rate=0.0)
            print("Target encoder re-synced.")

        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
            # Ensure target encoder is initialized correctly even if starting fresh
            model.target_encoder.update_ema(model.decoder_backbone, decay_rate=0.0)
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Checkpoint might be corrupted or incompatible.")
            print("Starting training from scratch or with potentially partially loaded state.")
            start_epoch = 0
            best_val_loss = float('inf')
            current_step = 0
            # Ensure target encoder is initialized correctly after error
            model.target_encoder.update_ema(model.decoder_backbone, decay_rate=0.0)

    else:
        if continue_training:
             print(f"Checkpoint file not found at {checkpoint_path} or continue_training=False.")
        print("Starting training from scratch.")
        # Initial sync of target encoder for fresh training
        model.target_encoder.update_ema(model.decoder_backbone, decay_rate=0.0)

    # --- Learning Rate Scheduler ---
    total_steps = hyperparams['num_epochs'] * hyperparams['steps_per_epoch']
    warmup_steps = hyperparams['warmup_steps']
    base_lr = hyperparams['learning_rate']
    min_lr = hyperparams['min_learning_rate']
    grad_clip = hyperparams['grad_clip']
    print(f"LR Scheduler: Cosine decay with warmup. Warmup steps: {warmup_steps}, Total steps: {total_steps}, Base LR: {base_lr}, Min LR: {min_lr}")

    def get_lr(step: int) -> float:
        """Calculates the learning rate for a given step using cosine decay with warmup."""
        # 1. Linear warmup phase
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps # Add 1 to step for 1-based counting? Or keep 0-based? Usually 0-based. step/warmup_steps
        # 2. Constant phase after decay (at min_lr)
        if step > total_steps: # Should ideally not happen if loop terminates correctly
            return min_lr
        # 3. Cosine decay phase
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        current_lr = min_lr + coeff * (base_lr - min_lr)
        return current_lr

    # --- Training Loop ---
    print(f"\n=== Starting Training (Epochs {start_epoch} to {hyperparams['num_epochs']}) ===")
    accumulation_steps = hyperparams['accumulation_steps']
    print(f"Gradient Accumulation Steps: {accumulation_steps}")

    # Optional: Setup Automatic Mixed Precision (AMP) for potential speedup on CUDA
    use_amp = (device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")


    for epoch in range(start_epoch, hyperparams['num_epochs']):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{hyperparams['num_epochs']} ---")
        model.train() # Set model to training mode for this epoch

        # Track losses within the epoch for logging
        epoch_total_loss_accum, epoch_jepa_loss_accum, epoch_lm_loss_accum = 0.0, 0.0, 0.0
        optimizer.zero_grad() # Zero gradients at the start of the epoch / after each optimizer step

        steps_in_epoch = hyperparams['steps_per_epoch']
        pbar = tqdm(range(steps_in_epoch), desc=f"Epoch {epoch+1} Training", ncols=100)

        for step_in_epoch in pbar:
            global_step = current_step # Track global step across epochs

            # --- Learning Rate Update ---
            # Set the learning rate for the current step *before* the optimizer step
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # --- Periodic Evaluation ---
            # Run evaluation at specified intervals (and at the beginning of training if step 0)
            if global_step % hyperparams['eval_interval'] == 0: # or global_step == 0: # Evaluate at start?
                eval_start_time = time.time()
                losses = estimate_loss(model, train_df, val_df, hyperparams, device)
                eval_duration = time.time() - eval_start_time
                print(f"\nStep {global_step} Evaluation Results (took {eval_duration:.2f}s):")
                print(f"  Train Loss -> Total: {losses['train_total']:.4f}, JEPA: {losses['train_jepa']:.4f}, LM: {losses['train_lm']:.4f}")
                print(f"  Val Loss   -> Total: {losses['val_total']:.4f}, JEPA: {losses['val_jepa']:.4f}, LM: {losses['val_lm']:.4f}")

                # Save the model if validation loss improved
                current_val_loss = losses['val_total']
                # Check if loss is valid (not NaN) before comparing
                if not math.isnan(current_val_loss) and current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    # Save a separate 'best' model checkpoint
                    best_save_path = checkpoint_path.replace('.pt', '_best.pt')
                    print(f"  New best validation loss: {best_val_loss:.4f}! Saving model to {best_save_path}...")
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch,
                        'current_step': global_step,
                        'val_loss': best_val_loss,
                        'hyperparams': hyperparams # Save hyperparams used for this checkpoint
                    }, best_save_path)
                else:
                     print(f"  Validation loss ({current_val_loss:.4f}) did not improve from best ({best_val_loss:.4f}).")

                # Ensure model is back in training mode after evaluation
                model.train()

            # --- Data Batch Preparation ---
            try:
                 # Get a batch of data with JEPA context/target masks
                 x, context_mask, target_spans_indices, attention_mask = prepare_batches_from_gsm8k(
                    train_df, hyperparams, device)
            except Exception as data_err:
                print(f"\nError preparing batch at step {global_step}: {data_err}. Skipping this step.")
                # If skipping a step that completes an accumulation cycle, reset grads
                # This logic assumes step_in_epoch is 0-based.
                is_accumulation_step = (step_in_epoch + 1) % accumulation_steps == 0
                if is_accumulation_step:
                     optimizer.zero_grad() # Reset gradients if skipping the optimizer step
                current_step += 1 # Increment step counter even if batch failed
                continue # Skip the rest of the loop for this step

            # --- Forward and Loss Calculation (within AMP context if enabled) ---
            # Autocast automatically chooses precision (float16/bfloat16) for compatible ops
            with torch.autocast(device_type=device if device != 'cpu' else 'cpu', dtype=torch.bfloat16 if device=='cuda' else torch.float32, enabled=use_amp):
                outputs = model(x, context_mask, target_spans_indices, attention_mask)
                loss_dict = model.compute_loss(outputs)
                # The loss should be computed *inside* the autocast context
                total_loss = loss_dict['total_loss']
                jepa_loss_item = loss_dict['jepa_loss'].item() # Get scalar value for logging
                lm_loss_item = loss_dict['lm_loss'].item()     # Get scalar value for logging

            # Check for NaN/Inf loss *before* backward pass
            if not torch.isfinite(total_loss):
                print(f"\nWarning: NaN or Inf loss detected at step {global_step}! Loss: {total_loss.item()}. Skipping step.")
                # If loss is invalid, clear gradients accumulated so far for this cycle and skip backward/optimizer step
                optimizer.zero_grad()
                current_step += 1
                continue # Skip to the next step

            # Accumulate losses for epoch-level average logging (use scalar values)
            epoch_total_loss_accum += total_loss.item() # Accumulate loss before scaling
            epoch_jepa_loss_accum += jepa_loss_item
            epoch_lm_loss_accum += lm_loss_item

            # --- Backward Pass ---
            # Scale the loss for gradient accumulation
            scaled_loss = total_loss / accumulation_steps
            # Use scaler to manage gradient scaling for AMP
            scaler.scale(scaled_loss).backward()

            # --- Optimizer Step (conditionally performed after accumulation) ---
            # Check if this step completes an accumulation cycle
            is_accumulation_step = (step_in_epoch + 1) % accumulation_steps == 0
            if is_accumulation_step:
                # --- Gradient Clipping and Optimizer Step ---
                # 1. Unscale gradients (required before clipping when using GradScaler)
                scaler.unscale_(optimizer)

                # 2. Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)

                # 3. Optimizer step (managed by scaler)
                # scaler.step() checks for inf/NaN gradients internally.
                # If gradients are invalid, it skips the optimizer step and adjusts scale.
                # If gradients are valid, it calls optimizer.step().
                scaler.step(optimizer)

                # 4. Update the GradScaler's scale factor for the next iteration
                scaler.update()

                # 5. Zero gradients for the next accumulation cycle
                optimizer.zero_grad()

                # --- Update Target Encoder (EMA) ---
                # Update the target encoder weights *after* the optimizer step
                # (uses the updated backbone weights)
                model.update_target_encoder()


                # --- Logging ---
                # Log performance metrics for the completed accumulation cycle
                avg_total_loss_epoch = epoch_total_loss_accum / (step_in_epoch + 1)
                avg_jepa_loss_epoch = epoch_jepa_loss_accum / (step_in_epoch + 1)
                avg_lm_loss_epoch = epoch_lm_loss_accum / (step_in_epoch + 1)

                pbar.set_description(f"E{epoch+1} S{global_step+1}/{total_steps}")
                pbar.set_postfix({
                    "LR": f"{lr:.2e}",
                    "AvgLoss": f"{avg_total_loss_epoch:.4f}",
                    # Use item() losses from the *last* step of the cycle for immediate feedback
                    "JEPA(last)": f"{jepa_loss_item:.4f}",
                    "LM(last)": f"{lm_loss_item:.4f}",
                    "Scale": f"{scaler.get_scale():.1f}" if use_amp else "N/A",
                    # Optionally track best val loss: "BestVal": f"{best_val_loss:.4f}"
                }, refresh=False) # Refresh=False might make tqdm smoother


            current_step += 1 # Increment global step counter after processing the step

        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        avg_total_loss_epoch = epoch_total_loss_accum / steps_in_epoch
        avg_jepa_loss_epoch = epoch_jepa_loss_accum / steps_in_epoch
        avg_lm_loss_epoch = epoch_lm_loss_accum / steps_in_epoch
        print(f"\nEpoch {epoch+1} completed in {epoch_duration:.2f}s.")
        print(f"  Average Epoch Loss -> Total: {avg_total_loss_epoch:.4f}, JEPA: {avg_jepa_loss_epoch:.4f}, LM: {avg_lm_loss_epoch:.4f}")


        # --- Generate Sample Text at Epoch End ---
        try:
            print("\nGenerating sample text at end of epoch...")
            model.eval() # Set to eval mode for generation
            # Use a shorter generation length for epoch-end samples
            sample_output = generate_token_by_token(
                model, hyperparams,
                prompt_text=hyperparams['start_prompt'], # Use the default prompt
                max_new_tokens=128, # Generate fewer tokens for quick sample
                device=device
            )
            # The function prints token-by-token, so we might just add separators
            print("-" * 30) # Separator after generation output
            # No need to print sample_output again if generate_token_by_token prints it.
            model.train() # Set back to train mode
        except Exception as e:
            print(f"Error during end-of-epoch sample generation: {e}")
            model.train() # Ensure model is back in train mode even if generation fails

        # --- Save End-of-Epoch Checkpoint ---
        # Save the latest model state, optimizer state, epoch number, etc.
        print(f"Saving end-of-epoch checkpoint to {checkpoint_path}...")
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch, # Save the completed epoch number
            'current_step': current_step, # Save the global step count
            'val_loss': best_val_loss, # Save the best validation loss seen so far
            'hyperparams': hyperparams # Save hyperparams with the checkpoint
        }, checkpoint_path)
        print("Checkpoint saved.")

    print("\n=== Training Complete ===")
    print(f"Final best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved at: {checkpoint_path} (latest) and {checkpoint_path.replace('.pt', '_best.pt')} (best validation)")


# ==========================================
# 14) Inference Implementation
# ==========================================
def inference(model_path: str, prompt_text: str, hyperparams_override: Optional[Dict[str, Any]] = None):
    """
    Runs inference using a trained T-JEPA DECODER model checkpoint.

    Loads the model structure based on hyperparameters saved in the checkpoint
    (or defaults if not found), loads the trained weights, and generates text
    token-by-token starting from the provided prompt.

    Args:
        model_path (str): Path to the model checkpoint file (.pt).
        prompt_text (str): The input prompt text to start generation from.
        hyperparams_override (Optional[Dict[str, Any]]): A dictionary of hyperparameters
                                                         to override those loaded from
                                                         the checkpoint or the defaults.
                                                         Useful for changing generation
                                                         parameters like 'top_p', 'temperature',
                                                         'generate_num_tokens'. Defaults to None.
    Returns:
        Optional[str]: The generated text output string, or None if loading failed.
                       The text is also printed to the console during generation.
    """
    device = get_device()

    # --- Load Checkpoint and Determine Hyperparameters ---
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at '{model_path}'")
        return None

    print(f"Loading model checkpoint from '{model_path}'...")
    try:
        # Load checkpoint onto the correct device
        checkpoint = torch.load(model_path, map_location=device)

        # Load hyperparams from checkpoint if available
        hyperparams_loaded = checkpoint.get('hyperparams', None)
        if hyperparams_loaded:
            print("  Hyperparameters found in checkpoint.")
            hyperparams = hyperparams_loaded # Start with checkpoint hyperparams
        else:
            print("  Warning: Hyperparameters not found in checkpoint. Using default values.")
            hyperparams = get_hyperparams() # Fallback to defaults

        # Apply overrides from the function argument
        if hyperparams_override:
            print(f"  Applying inference hyperparameter overrides: {hyperparams_override}")
            hyperparams.update(hyperparams_override)

        # Log the final effective hyperparameters being used for inference
        print(f"Effective hyperparameters for inference:")
        for key, val in hyperparams.items():
              print(f"    {key}: {val}")

    except Exception as e:
        print(f"Error reading checkpoint structure or hyperparameters: {e}")
        return None

    # --- Create Model Instance based on Hyperparameters ---
    print("\nInitializing model structure based on hyperparameters...")
    try:
        model = TJEPAModel(
            vocab_size=hyperparams['vocab_size'],
            embed_dim=hyperparams['embed_dim'],
            n_heads=hyperparams['n_heads'],
            n_layers=hyperparams['n_layers'],
            block_size=hyperparams['block_size'],
            # These are needed for structure but less critical for inference itself
            ema_decay=hyperparams.get('ema_decay', 0.999),
            lm_loss_weight=hyperparams.get('lm_loss_weight', 0.1),
            pad_token_id=hyperparams['pad_token']
        ).to(device)
    except KeyError as e:
         print(f"Error: Missing essential hyperparameter '{e}' in loaded/default config. Cannot build model.")
         return None
    except Exception as e_build:
        print(f"Error building model structure: {e_build}")
        return None

    # --- Load Model State Weights ---
    print("Loading model state weights...")
    try:
        model_state = checkpoint['model_state']
        # Use flexible loading (non-strict) to handle potential minor mismatches
        current_model_dict = model.state_dict()
        processed_state_dict = {}
        warned_keys_shape = set()
        warned_keys_missing = set()

        for k, v in model_state.items():
            new_k = k
            if k.startswith("context_encoder."): # Handle potential old naming
                new_k = k.replace("context_encoder.", "decoder_backbone.", 1)

            if new_k in current_model_dict:
                 if v.shape == current_model_dict[new_k].shape:
                    processed_state_dict[new_k] = v
                 else:
                     if new_k not in warned_keys_shape:
                         print(f"    Warning: Shape mismatch for key '{new_k}'. Checkpoint: {v.shape}, Model: {current_model_dict[new_k].shape}. Skipping.")
                         warned_keys_shape.add(new_k)
            # else: # Key from checkpoint not in current model (might be ok, e.g., removed layer)
            #     if new_k not in warned_keys_missing:
            #          print(f"    Info: Key '{new_k}' from checkpoint not found in current model structure. Skipping.")
            #          warned_keys_missing.add(new_k)

        missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)

        if missing_keys:
            print(f"  Info: Some weights were missing in the checkpoint and were initialized randomly: {missing_keys}")
        if unexpected_keys:
             print(f"  Info: Some weights from the checkpoint were not used by the current model: {unexpected_keys}")

        print("Model state loaded successfully.")
        # Print checkpoint details if available
        loaded_epoch = checkpoint.get('epoch', 'N/A')
        loaded_step = checkpoint.get('current_step', 'N/A')
        loaded_val_loss = checkpoint.get('val_loss', 'N/A')
        if loaded_val_loss != 'N/A': loaded_val_loss = f"{loaded_val_loss:.4f}" # Format loss
        print(f"  Checkpoint Source Details: Epoch={loaded_epoch}, Step={loaded_step}, Val Loss={loaded_val_loss}")

    except KeyError:
        print("Error: 'model_state' key not found in the checkpoint file.")
        print("Attempting inference with initialized model weights (will likely perform poorly).")
    except Exception as e:
        print(f"Error loading model state weights: {e}")
        print("Attempting inference with initialized model weights (will likely perform poorly).")

    # --- Run Generation ---
    model.eval() # Ensure model is in evaluation mode
    print(f"\n--- Generating Response ---")
    # Prompt is already passed as argument `prompt_text`

    # Use token-by-token generation for streaming output to console
    result_text = generate_token_by_token(
        model=model,
        hyperparams=hyperparams, # Pass the effective hyperparams
        prompt_text=prompt_text,
        # Use generation-specific hyperparams if overridden, otherwise from loaded/default
        max_new_tokens=hyperparams.get('generate_num_tokens', 512), # Use effective value
        device=device
    )
    # The `generate_token_by_token` function prints the output during generation.
    # We return the final string as well.

    return result_text


# ==========================================
# 15) Main Entry Point
# ==========================================
if __name__ == "__main__":
    print("=============================================")
    print(" T-JEPA MTL Decoder RoPE GSM8K Script ")
    print("=============================================")

    # --- Configuration ---
    # Load default hyperparameters initially to get paths, default modes etc.
    default_hyperparams = get_hyperparams()

    # Choose script mode: "train" or "inference"
    # Set this variable to switch between training and running inference.
    # MODE = "train"
    MODE = "inference"

    # --- Inference Specific Settings (only used if MODE="inference") ---
    # The prompt to use when running in inference mode.
    INFERENCE_PROMPT = "Janet has 3 apples. She buys 5 more bags of apples, and each bag contains 4 apples. She then gives away 7 apples to her friends. How many apples does Janet have left?"
    # Specify the path to the model checkpoint file for inference.
    # Often, this will be the '_best.pt' checkpoint saved during training.
    # Default path derived from hyperparams, adjusted to point to '_best.pt'.
    INFERENCE_MODEL_PATH = default_hyperparams['checkpoint_path'].replace('.pt', '_best.pt')
    # Optional: Override hyperparameters specifically for inference (e.g., generation length)
    # Example: {'generate_num_tokens': 50, 'top_p': 0.9}
    INFERENCE_OVERRIDES = {'generate_num_tokens': 512, 'temperature': 0.7, 'top_p': 0.85}

    # --- Execution ---
    if MODE == "train":
        print("\nSelected Mode: Training")
        # Get the 'continue_training' flag from the default hyperparameters
        continue_training_flag = default_hyperparams['continue_training']
        print(f"Continue training from checkpoint (if exists): {continue_training_flag}")
        # Start the training process
        train(continue_training=continue_training_flag)

    elif MODE == "inference":
        print("\nSelected Mode: Inference")
        print(f"Using Inference Prompt: \"{INFERENCE_PROMPT}\"")
        print(f"Attempting to load model from: {INFERENCE_MODEL_PATH}")
        if INFERENCE_OVERRIDES:
             print(f"Applying Inference Overrides: {INFERENCE_OVERRIDES}")

        # --- Check if the specified model file exists ---
        # Prioritize the 'best' model path.
        model_path_to_use = INFERENCE_MODEL_PATH
        if not os.path.exists(model_path_to_use):
             print(f"Warning: Best model path '{model_path_to_use}' not found.")
             # Fallback to the regular (latest) checkpoint path from hyperparams
             base_checkpoint_path = default_hyperparams['checkpoint_path']
             if os.path.exists(base_checkpoint_path):
                 print(f"Attempting to use the latest checkpoint path instead: '{base_checkpoint_path}'")
                 model_path_to_use = base_checkpoint_path
             else:
                 print(f"Error: Neither best model ('{INFERENCE_MODEL_PATH}') nor latest checkpoint ('{base_checkpoint_path}') found.")
                 print("Cannot run inference without a model file. Please train a model first or provide a valid path.")
                 exit(1) # Exit script if no suitable model file is found

        # Run the inference function
        generated_output = inference(
            model_path=model_path_to_use,
            prompt_text=INFERENCE_PROMPT,
            hyperparams_override=INFERENCE_OVERRIDES
            )

        # The inference function prints output, but we can print the final result again if needed
        # if generated_output is not None:
        #     print("\n--- Final Generated Text ---")
        #     print(generated_output)

    else:
        # Handle invalid mode selection
        print(f"Error: Unknown MODE '{MODE}'. Please choose 'train' or 'inference'.")
        exit(1)

    print("\nScript finished.")
