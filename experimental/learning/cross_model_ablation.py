"""
Cross-Model Hidden State Transfer Experiment with Multiple Alignment Methods

Tests whether hidden states from one LLM can condition another LLM.
Relevant for LatentWire's cross-model interlingua goal.

Models: Llama 3.1 8B (hidden_size=4096) ↔ Mistral 7B (hidden_size=4096)

Alignment Methods (Training-Free):
1. No alignment (baseline)
2. Procrustes (SVD-based rotation)
3. Centered Procrustes (rotation after centering)
4. Scaled Procrustes (rotation + scale)
5. L-Cross OLS (ordinary least squares)

Calibration Data: WikiText-2 (10,000 diverse sentences)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
from pathlib import Path

# Checkpoint directory for saving calibration matrices (outside runs/ to persist across experiments)
CHECKPOINT_DIR = Path("checkpoints/cross_model_ablation")

# ============================================================================
# Calibration Data Loading
# ============================================================================

def load_calibration_texts(num_samples=5000, min_length=50):
    """Load diverse calibration texts from WikiText-2"""
    print(f"Loading WikiText-2 calibration data ({num_samples} samples)...")

    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

        # Filter and sample
        calibration_texts = []
        for text in ds["text"]:
            text = text.strip()
            if len(text) >= min_length:
                calibration_texts.append(text)
                if len(calibration_texts) >= num_samples:
                    break

        print(f"✓ Loaded {len(calibration_texts)} calibration texts")
        return calibration_texts

    except Exception as e:
        print(f"Warning: Could not load WikiText-2: {e}")
        print("Falling back to default calibration texts...")
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "The Renaissance was a period of cultural rebirth in Europe.",
            "Quantum mechanics describes the behavior of matter at atomic scales.",
        ]


# ============================================================================
# Alignment Method Classes
# ============================================================================

class AlignmentMethod:
    """Base class for alignment methods"""

    def __init__(self):
        self.is_calibrated = False

    def calibrate(self, model_a, tokenizer_a, model_b, tokenizer_b, calibration_texts, device):
        """Compute alignment from calibration texts"""
        raise NotImplementedError

    def align(self, source_hidden):
        """Apply alignment to source hidden states"""
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class NoAlignment(AlignmentMethod):
    """No alignment - pass through source hidden states"""

    def calibrate(self, model_a, tokenizer_a, model_b, tokenizer_b, calibration_texts, device):
        self.is_calibrated = True

    def align(self, source_hidden):
        return source_hidden


class ProcrustesAlignment(AlignmentMethod):
    """Basic Procrustes: finds optimal rotation matrix W"""

    def __init__(self):
        super().__init__()
        self.W = None

    def calibrate(self, model_a, tokenizer_a, model_b, tokenizer_b, calibration_texts, device):
        # Create checkpoint filename based on method and direction
        direction = f"{model_a.config._name_or_path.split('/')[-1]}_to_{model_b.config._name_or_path.split('/')[-1]}"
        checkpoint_path = CHECKPOINT_DIR / f"procrustes_{direction}_{len(calibration_texts)}.pt"

        # Try to load from checkpoint
        if checkpoint_path.exists():
            print(f"  Loading Procrustes alignment from checkpoint: {checkpoint_path.name}")
            checkpoint = torch.load(checkpoint_path)
            self.W = checkpoint['W'].to(device)
            self.is_calibrated = True
            print(f"  ✓ Procrustes matrix loaded: {self.W.shape}")
            return

        print("  Computing Procrustes alignment (SVD-based rotation)...")

        all_source = []
        all_target = []

        for i, text in enumerate(calibration_texts):
            if i % 100 == 0:
                print(f"    Processing {i}/{len(calibration_texts)}...")

            inputs_a = tokenizer_a(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs_b = tokenizer_b(text, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs_a = model_a(**inputs_a, output_hidden_states=True)
                outputs_b = model_b(**inputs_b, output_hidden_states=True)

                hidden_a = outputs_a.hidden_states[-1]
                hidden_b = outputs_b.hidden_states[-1]

                # Handle different sequence lengths
                min_len = min(hidden_a.shape[1], hidden_b.shape[1])
                all_source.append(hidden_a[:, :min_len, :])
                all_target.append(hidden_b[:, :min_len, :])

        # Concatenate and flatten
        source_states = torch.cat([s.reshape(-1, s.shape[-1]) for s in all_source], dim=0)
        target_states = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_target], dim=0)

        # Compute W = U @ V.T from SVD
        H = (target_states.T @ source_states).float()
        eps = 1e-4
        H_reg = H + eps * torch.eye(H.shape[0], device=H.device)

        try:
            U, S, Vt = torch.linalg.svd(H_reg)
            self.W = (U @ Vt).to(source_states.dtype)
            print(f"  ✓ Procrustes matrix: {self.W.shape}")
        except:
            print("  Warning: SVD failed, using identity matrix")
            self.W = torch.eye(H.shape[0], device=device, dtype=source_states.dtype)

        self.is_calibrated = True

        # Save checkpoint
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({'W': self.W.cpu()}, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")

    def align(self, source_hidden):
        """Apply W @ source"""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before aligning")

        # W @ source: (D, D) @ (B, T, D).transpose -> (D, B*T).transpose -> (B*T, D) -> (B, T, D)
        flat = source_hidden.reshape(-1, source_hidden.shape[-1])
        aligned_flat = (self.W @ flat.T).T
        return aligned_flat.reshape(source_hidden.shape)


class CenteredProcrustesAlignment(AlignmentMethod):
    """Centered Procrustes: centers data before SVD"""

    def __init__(self):
        super().__init__()
        self.W = None
        self.source_mean = None
        self.target_mean = None

    def calibrate(self, model_a, tokenizer_a, model_b, tokenizer_b, calibration_texts, device):
        print("  Computing Centered Procrustes alignment (centers data before SVD)...")

        all_source = []
        all_target = []

        for i, text in enumerate(calibration_texts):
            if i % 100 == 0:
                print(f"    Processing {i}/{len(calibration_texts)}...")

            inputs_a = tokenizer_a(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs_b = tokenizer_b(text, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs_a = model_a(**inputs_a, output_hidden_states=True)
                outputs_b = model_b(**inputs_b, output_hidden_states=True)

                hidden_a = outputs_a.hidden_states[-1]
                hidden_b = outputs_b.hidden_states[-1]

                min_len = min(hidden_a.shape[1], hidden_b.shape[1])
                all_source.append(hidden_a[:, :min_len, :])
                all_target.append(hidden_b[:, :min_len, :])

        # Concatenate and flatten
        source_states = torch.cat([s.reshape(-1, s.shape[-1]) for s in all_source], dim=0)
        target_states = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_target], dim=0)

        # Compute means
        self.source_mean = source_states.mean(dim=0)
        self.target_mean = target_states.mean(dim=0)

        # Center data
        source_centered = source_states - self.source_mean
        target_centered = target_states - self.target_mean

        # Compute W on centered data
        H = (target_centered.T @ source_centered).float()
        eps = 1e-4
        H_reg = H + eps * torch.eye(H.shape[0], device=H.device)

        try:
            U, S, Vt = torch.linalg.svd(H_reg)
            self.W = (U @ Vt).to(source_states.dtype)
            print(f"  ✓ Centered Procrustes matrix: {self.W.shape}")
        except:
            print("  Warning: SVD failed, using identity matrix")
            self.W = torch.eye(H.shape[0], device=device, dtype=source_states.dtype)

        self.is_calibrated = True

    def align(self, source_hidden):
        """Apply (source - mean_s) @ W + mean_t"""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before aligning")

        flat = source_hidden.reshape(-1, source_hidden.shape[-1])
        centered = flat - self.source_mean
        aligned_flat = (self.W @ centered.T).T + self.target_mean
        return aligned_flat.reshape(source_hidden.shape)


class ScaledProcrustesAlignment(AlignmentMethod):
    """Scaled Procrustes: rotation + uniform scale"""

    def __init__(self):
        super().__init__()
        self.W = None
        self.scale = None

    def calibrate(self, model_a, tokenizer_a, model_b, tokenizer_b, calibration_texts, device):
        print("  Computing Scaled Procrustes alignment (rotation + scale)...")

        all_source = []
        all_target = []

        for i, text in enumerate(calibration_texts):
            if i % 100 == 0:
                print(f"    Processing {i}/{len(calibration_texts)}...")

            inputs_a = tokenizer_a(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs_b = tokenizer_b(text, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs_a = model_a(**inputs_a, output_hidden_states=True)
                outputs_b = model_b(**inputs_b, output_hidden_states=True)

                hidden_a = outputs_a.hidden_states[-1]
                hidden_b = outputs_b.hidden_states[-1]

                min_len = min(hidden_a.shape[1], hidden_b.shape[1])
                all_source.append(hidden_a[:, :min_len, :])
                all_target.append(hidden_b[:, :min_len, :])

        # Concatenate and flatten
        source_states = torch.cat([s.reshape(-1, s.shape[-1]) for s in all_source], dim=0)
        target_states = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_target], dim=0)

        # Compute rotation W
        H = (target_states.T @ source_states).float()
        eps = 1e-4
        H_reg = H + eps * torch.eye(H.shape[0], device=H.device)

        try:
            U, S, Vt = torch.linalg.svd(H_reg)
            self.W = (U @ Vt).to(source_states.dtype)
        except:
            print("  Warning: SVD failed, using identity matrix")
            self.W = torch.eye(H.shape[0], device=device, dtype=source_states.dtype)

        # Compute scale factor
        source_norm = torch.norm(source_states, dim=-1).mean()
        target_norm = torch.norm(target_states, dim=-1).mean()
        self.scale = target_norm / source_norm

        print(f"  ✓ Scaled Procrustes matrix: {self.W.shape}, scale: {self.scale:.4f}")
        self.is_calibrated = True

    def align(self, source_hidden):
        """Apply scale * (W @ source)"""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before aligning")

        flat = source_hidden.reshape(-1, source_hidden.shape[-1])
        aligned_flat = self.scale * (self.W @ flat.T).T
        return aligned_flat.reshape(source_hidden.shape)


class LCrossOLS(AlignmentMethod):
    """L-Cross Modulation: OLS-based linear transformation (from Paper 2)"""

    def __init__(self):
        super().__init__()
        self.T = None

    def calibrate(self, model_a, tokenizer_a, model_b, tokenizer_b, calibration_texts, device):
        print("  Computing L-Cross OLS alignment (least squares)...")

        all_source = []
        all_target = []

        for i, text in enumerate(calibration_texts):
            if i % 100 == 0:
                print(f"    Processing {i}/{len(calibration_texts)}...")

            inputs_a = tokenizer_a(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs_b = tokenizer_b(text, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs_a = model_a(**inputs_a, output_hidden_states=True)
                outputs_b = model_b(**inputs_b, output_hidden_states=True)

                hidden_a = outputs_a.hidden_states[-1]
                hidden_b = outputs_b.hidden_states[-1]

                min_len = min(hidden_a.shape[1], hidden_b.shape[1])
                all_source.append(hidden_a[:, :min_len, :])
                all_target.append(hidden_b[:, :min_len, :])

        # Concatenate and flatten
        source_states = torch.cat([s.reshape(-1, s.shape[-1]) for s in all_source], dim=0)
        target_states = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_target], dim=0)

        # Solve: T = argmin ||target - T @ source||² using least squares
        # T = target @ source.pinv() OR use torch.linalg.lstsq
        # Convert to float32 for linear algebra operations (don't support half precision)
        # Move to CPU to avoid OOM (lstsq needs ~11GB extra memory)
        print("  Moving data to CPU for OLS computation (avoiding OOM)...")
        source_f32 = source_states.float().cpu()
        target_f32 = target_states.float().cpu()

        try:
            # Using lstsq: solve X @ T.T = Y for T.T, where X=source, Y=target
            solution = torch.linalg.lstsq(source_f32, target_f32)
            self.T = solution.solution.T.to(device).to(source_states.dtype)
            print(f"  ✓ L-Cross OLS matrix: {self.T.shape}")
        except Exception as e:
            print(f"  Warning: lstsq failed ({e}), using pseudo-inverse")
            self.T = (target_f32.T @ torch.pinverse(source_f32.T)).to(device).to(source_states.dtype)

        self.is_calibrated = True

    def align(self, source_hidden):
        """Apply T @ source"""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before aligning")

        flat = source_hidden.reshape(-1, source_hidden.shape[-1])
        aligned_flat = (self.T @ flat.T).T
        return aligned_flat.reshape(source_hidden.shape)


# ============================================================================
# Generation Functions
# ============================================================================

def generate_baseline(model, tokenizer, prompt, max_new_tokens):
    """Standard generation with MPS compatibility"""
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # MPS has issues with model.generate(), so we use manual generation
    device = model.device
    if str(device).startswith("mps"):
        # Manual generation for MPS
        input_ids = inputs.input_ids
        generated_ids = input_ids[0].tolist()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                curr_input = torch.tensor([generated_ids]).to(device)
                outputs = model(curr_input)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()

                if next_token_id == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id)

        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        # Use built-in generate for CUDA/CPU
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_cross_model(
    model_a, tokenizer_a,
    model_b, tokenizer_b,
    prompt, max_new_tokens,
    alignment_method=None
):
    """
    Generate using Model A's hidden states as input to Model B.

    Args:
        alignment_method: AlignmentMethod instance (or None for no alignment)
    """
    # Tokenize with Model A's tokenizer
    inputs_a = tokenizer_a(prompt, return_tensors="pt").to(model_a.device)

    # Get hidden states from Model A (no generation)
    with torch.no_grad():
        outputs_a = model_a.model(**inputs_a, output_hidden_states=True)
        hidden_states_a = outputs_a.hidden_states[-1]  # Last layer
        print(f"Model A hidden states: {hidden_states_a.shape}")

    # Verify dimensions
    model_a_dim = hidden_states_a.shape[-1]
    model_b_dim = model_b.config.hidden_size
    print(f"Model A dim: {model_a_dim}, Model B dim: {model_b_dim}")

    if model_a_dim != model_b_dim:
        raise ValueError(f"Dimension mismatch! {model_a_dim} != {model_b_dim}")

    # Apply alignment if provided
    if alignment_method is not None:
        print(f"  Applying {alignment_method}...")
        hidden_states_b = alignment_method.align(hidden_states_a)
    else:
        hidden_states_b = hidden_states_a

    # Generate with Model B from Model A's hidden states (with KV cache)
    generated_ids = []
    past_key_values = None
    current_hidden = hidden_states_b

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass with KV cache
            if past_key_values is None:
                # First step: process all initial hidden states
                outputs_b = model_b.model(
                    inputs_embeds=current_hidden,
                    past_key_values=None,
                    use_cache=True,
                    output_hidden_states=True
                )
            else:
                # Subsequent steps: only process new token embedding
                outputs_b = model_b.model(
                    inputs_embeds=current_hidden[:, -1:, :],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )

            # Get logits and select next token (greedy)
            logits = model_b.lm_head(outputs_b.hidden_states[-1])
            next_token_id = torch.argmax(logits[0, -1, :]).item()

            # Store KV cache
            past_key_values = outputs_b.past_key_values

            # Check for EOS
            if next_token_id == tokenizer_b.eos_token_id:
                break

            generated_ids.append(next_token_id)

            # Get embedding for next token and concatenate
            next_embedding = model_b.model.embed_tokens(torch.tensor([[next_token_id]]).to(model_b.device))
            current_hidden = torch.cat([current_hidden, next_embedding], dim=1)

    # Decode
    generated_text = tokenizer_b.decode(generated_ids, skip_special_tokens=True)
    return prompt + generated_text


# ============================================================================
# Main Experiment
# ============================================================================

def cross_model_experiment():
    """
    Test all alignment methods for cross-model hidden state transfer.
    """

    # Load models
    print("Loading models...")
    llama_model_id = "meta-llama/Llama-3.1-8B"
    mistral_model_id = "mistralai/Mistral-7B-v0.1"

    # Detect device and choose loading strategy
    if torch.cuda.is_available():
        print("Using device: cuda")
        device = "cuda"
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
    elif torch.backends.mps.is_available():
        print("Using device: mps")
        device = "mps"
        load_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True
        }
    else:
        print("⚠️ Using device: cpu (will be very slow!)")
        device = "cpu"
        load_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True
        }

    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    print(f"Loading {llama_model_id}...")
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, **load_kwargs)
    if device != "cuda":
        llama_model = llama_model.to(device)

    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)
    print(f"Loading {mistral_model_id}...")
    mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_id, **load_kwargs)
    if device != "cuda":
        mistral_model = mistral_model.to(device)

    print("✓ Models loaded successfully!")
    print(f"Llama device: {next(llama_model.parameters()).device}")
    print(f"Mistral device: {next(mistral_model.parameters()).device}")

    # Load calibration data
    calibration_texts = load_calibration_texts(num_samples=5000)

    # Test prompt
    prompt = "The future of artificial intelligence is"
    max_new_tokens = 50

    # Baseline experiments
    print("\n" + "="*60)
    print("BASELINE EXPERIMENTS")
    print("="*60)

    print("\n=== Llama 3.1 8B Alone ===")
    llama_only = generate_baseline(llama_model, llama_tokenizer, prompt, max_new_tokens)
    print(llama_only)

    print("\n=== Mistral 7B Alone ===")
    mistral_only = generate_baseline(mistral_model, mistral_tokenizer, prompt, max_new_tokens)
    print(mistral_only)

    # Sanity check: same model
    print("\n" + "="*60)
    print("SANITY CHECKS (Same Model Transfer)")
    print("="*60)

    print("\n=== Llama 3.1 8B → Llama 3.1 8B ===")
    llama_to_llama = generate_cross_model(
        llama_model, llama_tokenizer,
        llama_model, llama_tokenizer,
        prompt, max_new_tokens,
        alignment_method=None
    )
    print(llama_to_llama)

    print("\n=== Mistral 7B → Mistral 7B ===")
    mistral_to_mistral = generate_cross_model(
        mistral_model, mistral_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens,
        alignment_method=None
    )
    print(mistral_to_mistral)

    # Cross-model experiments with all alignment methods
    print("\n" + "="*60)
    print("CROSS-MODEL TRANSFER EXPERIMENTS")
    print("="*60)

    # Initialize alignment methods
    alignment_methods = [
        ("No Alignment", NoAlignment()),
        ("Procrustes", ProcrustesAlignment()),
        ("Centered Procrustes", CenteredProcrustesAlignment()),
        ("Scaled Procrustes", ScaledProcrustesAlignment()),
        ("L-Cross OLS", LCrossOLS()),
    ]

    # Calibrate all methods
    print("\n" + "="*60)
    print("CALIBRATING ALIGNMENT METHODS")
    print("="*60)

    for name, method in alignment_methods:
        print(f"\n{name}:")
        method.calibrate(llama_model, llama_tokenizer, mistral_model, mistral_tokenizer, calibration_texts, device)

    # Test all methods: Llama → Mistral
    print("\n" + "="*60)
    print("Llama 3.1 8B → Mistral 7B (All Alignment Methods)")
    print("="*60)

    for name, method in alignment_methods:
        print(f"\n=== {name} ===")
        result = generate_cross_model(
            llama_model, llama_tokenizer,
            mistral_model, mistral_tokenizer,
            prompt, max_new_tokens,
            alignment_method=method if name != "No Alignment" else None
        )
        print(result)

    # Test all methods: Mistral → Llama
    print("\n" + "="*60)
    print("Mistral 7B → Llama 3.1 8B (All Alignment Methods)")
    print("="*60)

    # Need to recalibrate in reverse direction
    print("\nRecalibrating for reverse direction (Mistral → Llama)...")
    reverse_methods = [
        ("No Alignment", NoAlignment()),
        ("Procrustes", ProcrustesAlignment()),
        ("Centered Procrustes", CenteredProcrustesAlignment()),
        ("Scaled Procrustes", ScaledProcrustesAlignment()),
        ("L-Cross OLS", LCrossOLS()),
    ]

    for name, method in reverse_methods:
        print(f"\n{name}:")
        method.calibrate(mistral_model, mistral_tokenizer, llama_model, llama_tokenizer, calibration_texts, device)

    for name, method in reverse_methods:
        print(f"\n=== {name} ===")
        result = generate_cross_model(
            mistral_model, mistral_tokenizer,
            llama_model, llama_tokenizer,
            prompt, max_new_tokens,
            alignment_method=method if name != "No Alignment" else None
        )
        print(result)

    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)


# Run experiments
if __name__ == "__main__":
    cross_model_experiment()
