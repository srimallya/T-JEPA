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
MODEL_PATH = "t_jepa_mtl_decoder_rope_bs1024_checkpoint.pt" # Or your specific checkpoint path
PROMPT_TEXT = "what is five plus two?"
NUM_VOTES = 8 # K: Number of samples per token
MAX_NEW_TOKENS = 2048 # L: Maximum generation length
TEMPERATURE = 0.7 # Sampling temperature
TOP_P = 0.9 # Top-p nucleus sampling
SYSTEM_PROMPT = """Consider this math problem. Think step by step and provide your reasoning between <think> </think> tags, then give your final answer between <answer> </answer> tags."""
THINK_TAG_START = "<think>" # Used to start generation after the prompt

# ==========================================
# Model Definitions (Copied from training script)
# ==========================================

# --- 1) Hyperparameters Default ---
# These will be OVERRIDDEN by checkpoint if available
def get_default_hyperparams():
    # Provide sensible defaults in case checkpoint doesn't contain hyperparams
    return {
        'vocab_size': 256, 'embed_dim': 512, 'n_heads': 8, 'n_layers': 12,
        'block_size': 1024, 'ema_decay': 0.999, 'lm_loss_weight': 0.1,
        'bos_token': 254, 'eos_token': 255, 'pad_token': 0,
        'top_p': 0.8, # Default for generation if not overridden
        # JEPA params (not strictly needed for inference, but part of model structure)
        'context_span_ratio': 0.6, 'target_span_ratio': 0.2,
        'num_target_spans': 8, 'min_span_length': 32,
        # Tags (used in helper functions)
        'thinking_tag': "<think>", 'thinking_end_tag': "</think>",
        'answer_tag': "<answer>", 'answer_end_tag': "</answer>",
        'system_prompt': """Consider this math problem. Think step by step and provide your reasoning between <think> </think> tags, then give your final answer between <answer> </answer> tags."""
    }

# --- 2) RoPE ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
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
        if seq_len > self.max_seq_len:
             print(f"Warning: RoPE seq_len {seq_len} > max_seq_len {self.max_seq_len}. Clamping.")
             seq_len = self.max_seq_len
            # raise ValueError(f"RoPE sequence length {seq_len} exceeds precomputed max {self.max_seq_len}") # Clamp instead

        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- 3) Attention ---
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

        if self.use_rope and self.is_self_attention:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.rotary_emb = None

        self.attn_dropout = nn.Dropout(0.1)
        self.out_dropout = nn.Dropout(0.1)
        self.register_buffer("causal_mask_cache", None, persistent=False)

    def _get_causal_mask(self, T, device):
        if self.causal_mask_cache is None or self.causal_mask_cache.shape[-1] < T:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
            self.causal_mask_cache = mask
        return self.causal_mask_cache[:T, :T].to(device=device)

    def forward(self, x, attn_mask=None, key_value_states=None, is_causal=False):
        B, T, C = x.size()
        is_cross_attn = key_value_states is not None
        use_rope_for_this_pass = self.use_rope and self.is_self_attention and not is_cross_attn and self.rotary_emb is not None

        q = self.q_proj(x)
        if is_cross_attn:
            T_k = key_value_states.size(1)
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
            is_causal = False
        else:
            T_k = T
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        if use_rope_for_this_pass:
            cos, sin = self.rotary_emb(T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            scaling_factor = 1.0
        else:
            scaling_factor = 1.0 / math.sqrt(self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling_factor

        final_mask_bool = None
        if attn_mask is not None:
            if attn_mask.dim() == 2: padding_mask_bool = ~attn_mask.bool().unsqueeze(1).unsqueeze(2)
            elif attn_mask.dim() == 4: padding_mask_bool = ~attn_mask.bool()
            else: raise ValueError(f"Unsupported attn_mask dimension: {attn_mask.dim()}")
            final_mask_bool = padding_mask_bool

        if self.is_self_attention and is_causal:
            causal_mask_bool = self._get_causal_mask(T, x.device).unsqueeze(0).unsqueeze(0)
            if final_mask_bool is not None: final_mask_bool = final_mask_bool | causal_mask_bool
            else: final_mask_bool = causal_mask_bool

        if final_mask_bool is not None:
             scores = scores.masked_fill(final_mask_bool, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_dropout(self.out_proj(attn_output))

# --- 4) Decoder Block ---
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attention = ImprovedAttention(embed_dim, n_heads, is_self_attention=True, use_rope=True, max_seq_len=max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = 4 * embed_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, is_causal=True):
        residual = x
        x_norm = self.ln1(x)
        attn_output = self.self_attention(x_norm, attn_mask=attention_mask, is_causal=is_causal)
        x = residual + self.dropout(attn_output)
        residual = x
        x_norm = self.ln2(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)
        return x

# --- 5) JEPA Predictor Block (Needed for model structure, not used in generation logic) ---
class JEPAPredictorBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, max_seq_len=2048):
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

    def forward(self, x, decoder_output, self_attention_mask=None, cross_attention_mask=None):
        # --- Self-Attention ---
        residual = x
        attn_output = self.self_attention(self.ln1(x), attn_mask=self_attention_mask, is_causal=True)
        x = residual + self.dropout(attn_output)
        # --- Cross-Attention ---
        residual = x
        cross_attn_output = self.cross_attention(
            self.ln_cross_attn_query(x),
            attn_mask=cross_attention_mask,
            key_value_states=self.ln_cross_attn_kv(decoder_output)
        )
        x = residual + self.dropout(cross_attn_output)
        # --- Feed-Forward ---
        residual = x
        ff_output = self.feed_forward(self.ln3(x))
        x = residual + self.dropout(ff_output)
        return x

# --- 6) Backbone Decoder ---
class BackboneDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, dropout=0.1, max_seq_len=block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02); torch.nn.init.zeros_(module.bias) if module.bias is not None else None
        elif isinstance(module, nn.Embedding): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm): torch.nn.init.zeros_(module.bias); torch.nn.init.ones_(module.weight)

    def forward(self, x, attention_mask=None, is_causal=True):
        B, T = x.size()
        # assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}" # Allow longer during generation cropping
        token_emb = self.token_embedding(x)
        x = self.dropout(token_emb)
        for block in self.blocks: x = block(x, attention_mask=attention_mask, is_causal=is_causal)
        x = self.ln_f(x)
        return x

# --- 7) JEPA Predictor (Needed for model structure, not used in generation logic) ---
class JEPAPredictor(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, block_size):
        super().__init__()
        self.block_size = block_size
        predictor_layers = n_layers
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([
            JEPAPredictorBlock(embed_dim, n_heads, max_seq_len=block_size)
            for _ in range(predictor_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02); torch.nn.init.zeros_(module.bias) if module.bias is not None else None
        elif isinstance(module, nn.LayerNorm): torch.nn.init.zeros_(module.bias); torch.nn.init.ones_(module.weight)

    def forward(self, decoder_output_causal, target_spans_indices, context_mask, attention_mask):
        # This forward function is complex and primarily for training JEPA loss.
        # It's not directly called during standard autoregressive generation.
        # We include the structure for model loading compatibility.
        pass # Not needed for SR-ABI inference logic

# --- 8) Target Encoder (Needed for model structure, not used in generation logic) ---
class TargetEncoder(nn.Module):
    def __init__(self, backbone_decoder, ema_decay=0.999):
        super().__init__()
        self.encoder = copy.deepcopy(backbone_decoder)
        self.ema_decay = ema_decay
        for param in self.encoder.parameters(): param.requires_grad = False

    @torch.no_grad()
    def update_ema(self, backbone_decoder, decay_rate=None): pass # Not needed for inference

    @torch.no_grad()
    def forward(self, x, attention_mask=None): pass # Not needed for inference

# --- 9) Complete T-JEPA Model ---
class TJEPAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, block_size, ema_decay=0.999, lm_loss_weight=0.1, pad_token_id=0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.lm_loss_weight = lm_loss_weight
        self.block_size = block_size

        self.decoder_backbone = BackboneDecoder(vocab_size, embed_dim, n_heads, n_layers, block_size)
        # Predictor and TargetEncoder needed for state_dict loading compatibility
        self.predictor = JEPAPredictor(embed_dim, n_heads, n_layers, block_size)
        self.target_encoder = TargetEncoder(self.decoder_backbone, ema_decay)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.decoder_backbone.token_embedding.weight = self.lm_head.weight # Weight tying

    # Forward methods related to training (JEPA loss) are omitted for inference clarity
    # We only need the backbone and lm_head for generation.

# ==========================================
# Helper Functions for Tokenization
# ==========================================
def _encode(text: str, bos_token: int) -> List[int]:
    """Encodes text to byte tokens, adding BOS."""
    return [bos_token] + [b for b in text.encode('utf-8', errors='replace')]

def _decode(tokens: List[int], bos_token: int, eos_token: int, pad_token: int) -> str:
    """Decodes byte tokens to text, removing special tokens."""
    try:
        # Find EOS if present and truncate
        eos_pos = tokens.index(eos_token) if eos_token in tokens else -1
        if eos_pos != -1:
            tokens = tokens[:eos_pos]

        # Filter out BOS and PAD, then decode
        filtered_bytes = bytes([tok for tok in tokens if tok != bos_token and tok != pad_token])
        return filtered_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Decoding error: {e}")
        return f"[Decoding Error] Raw bytes: {bytes(tokens)}"

# ==========================================
# SR-ABI Inference Function
# ==========================================
@torch.no_grad()
def generate_sr_abi(
    model: TJEPAModel,
    prompt_text: str,
    num_votes: int,           # K
    max_new_tokens: int,      # L
    temperature: float,       # Part of Theta
    top_p: float,             # Part of Theta
    hyperparams: Dict[str, Any], # Contains BOS, EOS, PAD IDs etc.
    device: str
) -> str:
    """
    Generates text using State-Resetting Agreement-Based Inference (SR-ABI).
    """
    model.eval()
    bos_token = hyperparams['bos_token']
    eos_token = hyperparams['eos_token']
    pad_token = hyperparams['pad_token']
    block_size = hyperparams['block_size']

    # --- Initialization ---
    # a. Tokenize prompt (Including system prompt and starting tag)
    full_prompt = f"{hyperparams.get('system_prompt', '')}\n\nProblem: {prompt_text}\n\n{THINK_TAG_START}"
    prompt_tokens = _encode(full_prompt, bos_token)
    # b. Initialize current full sequence S
    S_list = prompt_tokens[:] # List of token IDs
    # c. Initialize generated sequence G
    G_list = [] # List of token IDs (only the generated part)

    print(f"\n--- Starting SR-ABI Generation (K={num_votes}) ---")
    print(f"Prompt:\n{full_prompt}", end="", flush=True)

    # --- Token Generation Loop ---
    for i in range(max_new_tokens):
        # --- a. Vote Collection ---
        votes = Counter()
        S_tensor = torch.tensor([S_list], dtype=torch.long, device=device) # Add batch dim [1, T]

        # Crop context if it exceeds block size for the forward pass
        S_cond = S_tensor if S_tensor.size(1) <= block_size else S_tensor[:, -block_size:]
        seq_len = S_cond.size(1)

        # Create attention mask for the current sequence
        attention_mask = (S_cond != pad_token).float().to(device) # [1, T_cond]

        for j in range(num_votes):
            # --- i.1 & i.2: Reset State & Re-evaluate Context ---
            # This is achieved by running the forward pass on the *full* current
            # sequence S_cond. The model calculates state (KV cache) from scratch.
            decoder_output = model.decoder_backbone(
                S_cond,
                attention_mask=attention_mask,
                is_causal=True # Standard causal generation
            ) # [1, T_cond, C]

            # Get logits for the *next* token prediction (using the last token's embedding)
            logits = model.lm_head(decoder_output[:, -1, :])  # [1, C] -> [1, V]

            # --- i.3: Sample Candidate ---
            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                 logits = logits / temperature

            # Apply top-p (nucleus) sampling
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            candidate_token_tensor = torch.multinomial(probs, num_samples=1) # [1, 1]
            candidate_token = candidate_token_tensor.item()

            # --- i.4: Record Vote ---
            votes[candidate_token] += 1

        # --- b. Agreement (Majority Vote) ---
        if not votes:
            print("\nWarning: No votes collected, stopping generation.")
            break # Should not happen if num_votes >= 1

        # Get the token with the most votes. Tie-breaking: implicitly handled by most_common
        # (returns items in order of first appearance among ties if counts are equal)
        winning_token, vote_count = votes.most_common(1)[0]
        # Optional: print vote distribution for debugging
        # print(f"\nVotes (Step {i+1}): {votes}")
        # print(f"Winner: {winning_token} ({vote_count}/{num_votes})")

        # --- c. Check for Termination ---
        if winning_token == eos_token:
            print("<EOS>", flush=True)
            break

        # --- d. Append Token ---
        G_list.append(winning_token)
        S_list.append(winning_token)

        # --- Print the winning token ---
        # Attempt to decode the single winning token for streaming output
        try:
            print(bytes([winning_token]).decode('utf-8', errors='replace'), end="", flush=True)
        except UnicodeDecodeError:
            print("<?>", end="", flush=True) # Placeholder for partial UTF-8 chars

        # Optional: Small delay for readability
        # time.sleep(0.01)

    print("\n--- Generation Complete ---")

    # --- Finalization ---
    # a. Detokenize the generated sequence G
    output_text = _decode(G_list, bos_token, eos_token, pad_token)

    return full_prompt + output_text # Return prompt + generated text


# ==========================================
# Model Loading Function
# ==========================================
def load_model_for_inference(model_path: str, device: str) -> Tuple[Optional[TJEPAModel], Optional[Dict[str, Any]]]:
    """Loads the TJEPAModel from a checkpoint for inference."""
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return None, None

    print(f"Loading model checkpoint from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        return None, None

    # Load hyperparams from checkpoint or use defaults
    hyperparams_loaded = checkpoint.get('hyperparams', None)
    if hyperparams_loaded:
        print("Using hyperparameters loaded from checkpoint.")
        # Merge with defaults to ensure all needed keys exist
        hyperparams = get_default_hyperparams()
        hyperparams.update(hyperparams_loaded) # Loaded values override defaults
    else:
        print("Warning: Hyperparameters not found in checkpoint, using default values.")
        hyperparams = get_default_hyperparams()

    print(f"Effective Hyperparameters: {hyperparams}")

    # Create model instance based on loaded/effective hyperparams
    try:
        model = TJEPAModel(
            vocab_size=hyperparams['vocab_size'], embed_dim=hyperparams['embed_dim'],
            n_heads=hyperparams['n_heads'], n_layers=hyperparams['n_layers'],
            block_size=hyperparams['block_size'], ema_decay=hyperparams['ema_decay'],
            lm_loss_weight=hyperparams['lm_loss_weight'], pad_token_id=hyperparams['pad_token']
        ).to(device)
    except KeyError as e:
         print(f"Error: Missing hyperparameter '{e}' needed to build the model structure.")
         return None, None
    except Exception as e:
        print(f"Error creating model instance: {e}")
        return None, None

    # Load model state dictionary
    try:
        model_state = checkpoint['model_state']
        # Flexible loading
        current_model_dict = model.state_dict()
        processed_state_dict = {}
        warned_keys = set()
        loaded_keys_count = 0
        for k, v in model_state.items():
            new_k = k # Handle potential renames if needed in the future
            # Example rename: if k.startswith("old_prefix."): new_k = k.replace("old_prefix.", "new_prefix.", 1)
            if new_k in current_model_dict:
                if v.shape == current_model_dict[new_k].shape:
                    processed_state_dict[new_k] = v
                    loaded_keys_count += 1
                else:
                    if new_k not in warned_keys:
                        print(f"Warning: Shape mismatch for key '{new_k}'. Checkpoint: {v.shape}, Model: {current_model_dict[new_k].shape}. Skipping.")
                        warned_keys.add(new_k)
            # else:
            #     if k not in warned_keys and new_k not in warned_keys:
            #         print(f"Warning: Key '{k}' (mapped to '{new_k}') not found in current model. Skipping.")
            #         warned_keys.add(k); warned_keys.add(new_k)

        missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)
        if missing_keys: print(f"  Info: Missing keys in final state_dict load: {missing_keys}")
        if unexpected_keys: print(f"  Info: Unexpected keys found in checkpoint but not used: {unexpected_keys}")
        print(f"Model state loaded successfully ({loaded_keys_count} tensors loaded).")
        loaded_epoch = checkpoint.get('epoch', -1); loaded_step = checkpoint.get('current_step', -1)
        val_loss = checkpoint.get('val_loss', 'N/A')
        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
        print(f"  Checkpoint details: Epoch {loaded_epoch}, Step {loaded_step}, Val Loss {val_loss_str}")

    except Exception as e:
        print(f"Error loading model state weights: {e}")
        print("Attempting inference with potentially uninitialized weights.")

    return model, hyperparams

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # --- Setup Device ---
    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model, hyperparams = load_model_for_inference(MODEL_PATH, device)

    if model and hyperparams:
        # --- Run SR-ABI Generation ---
        start_time = time.time()
        generated_text = generate_sr_abi(
            model=model,
            prompt_text=PROMPT_TEXT,
            num_votes=NUM_VOTES,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            hyperparams=hyperparams,
            device=device
        )
        end_time = time.time()

        print("\n\n--- Final Output ---")
        print(generated_text)
        print(f"\nGeneration took {end_time - start_time:.2f} seconds.")
    else:
        print("Failed to load model. Exiting.")
