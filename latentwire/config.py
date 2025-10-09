# latentwire/config.py
"""
Milestone 1: Config Schema & Default Toggles

Centralized configuration system using dataclasses for type safety and validation.
All feature toggles default to False; encoder training defaults to True.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """LLM model configuration."""
    llama_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    qwen_id: str = "Qwen/Qwen2-0.5B-Instruct"
    llama_device_map: Optional[str] = None
    qwen_device_map: Optional[str] = None
    llama_devices: Optional[str] = None
    qwen_devices: Optional[str] = None
    models: str = "llama,qwen"  # Comma-separated subset
    load_4bit: bool = False
    sequential_models: bool = False
    gpu_mem_gib: float = 78.0

    # Teacher models for KD
    teacher_llama_id: Optional[str] = None
    teacher_qwen_id: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset and sampling configuration."""
    dataset: str = "hotpot"
    hotpot_config: str = "fullwiki"
    samples: int = 128
    epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 1
    max_answer_tokens: int = 32

    # Reproducibility
    seed: int = 42
    data_seed: int = 42


@dataclass
class EncoderConfig:
    """Interlingua encoder configuration."""
    # Architecture
    encoder_type: str = "byte"  # byte, simple-st, stq
    latent_len: int = 8
    latent_shared_len: Optional[int] = None
    latent_private_len: int = 0
    d_z: int = 256
    max_bytes: int = 512

    # Encoder-specific options
    encoder_use_chat_template: bool = False
    encoder_backbone: Optional[str] = None
    hf_encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_enc_tokens: int = 1024

    # Trainability (NOTE: defaults to True per Milestone 1 spec)
    train_encoder: bool = True
    freeze_encoder: bool = False  # Legacy flag, overrides train_encoder if True


@dataclass
class AdapterConfig:
    """Adapter module configuration."""
    adapter_hidden_mult: int = 2
    adapter_dropout: float = 0.0
    adapter_colorize: bool = False
    adapter_metadata: bool = True
    adapter_freeze_scale: bool = False

    # Regularization
    scale_l2: float = 0.05
    adapter_rms_l2: float = 0.0


@dataclass
class FeatureToggles:
    """Optional features - all default to False per Milestone 1 spec."""

    # PEFT methods
    use_lora: bool = False
    use_prefix: bool = False

    # Advanced conditioning
    use_deep_prefix: bool = False
    use_latent_adapters: bool = False
    use_coprocessor: bool = False
    use_gist_head: bool = False

    # Refinement
    use_latent_refiner: bool = False


@dataclass
class LoRAConfig:
    """LoRA adapter configuration (only used if FeatureToggles.use_lora=True)."""
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_firstN: Optional[int] = None
    lora_target_modules: str = "auto"


@dataclass
class PrefixTuningConfig:
    """Prefix-tuning configuration (only used if FeatureToggles.use_prefix=True)."""
    prefix_tokens: int = 16
    prefix_projection: bool = False
    peft_prefix_all_layers: str = "yes"


@dataclass
class DeepPrefixConfig:
    """Deep prefix configuration (only used if FeatureToggles.use_deep_prefix=True)."""
    deep_prefix_len: Optional[int] = None  # Defaults to latent_shared_len
    deep_prefix_dropout: float = 0.1


@dataclass
class LatentAdapterConfig:
    """Multi-depth latent adapters (only used if FeatureToggles.use_latent_adapters=True)."""
    latent_adapter_layers: str = "8,16,24"
    latent_adapter_heads: int = 8
    latent_adapter_dropout: float = 0.1


@dataclass
class CoprocessorConfig:
    """Latent coprocessor configuration (only used if FeatureToggles.use_coprocessor=True)."""
    coprocessor_len: int = 1
    coprocessor_width: int = 256
    coprocessor_dropout: float = 0.1
    coprocessor_kv_scale: float = 0.8
    coprocessor_pool: str = "mean"  # mean, first, max
    coprocessor_heads: str = ""     # optional comma-separated override per layer


@dataclass
class GistConfig:
    """Gist reconstruction head (only used if FeatureToggles.use_gist_head=True)."""
    gist_target_len: int = 48
    gist_hidden: int = 512
    gist_layers: int = 2
    gist_dropout: float = 0.1
    gist_weight: float = 0.0
    gist_mask_prob: float = 0.15


@dataclass
class LatentRefinerConfig:
    """Latent refiner (only used if FeatureToggles.use_latent_refiner=True)."""
    latent_refiner_layers: int = 0
    latent_refiner_heads: int = 4


@dataclass
class LossWeights:
    """Training loss weights and KD configuration."""
    # First-token objectives
    first_token_ce_weight: float = 0.5
    first_token_ce_schedule: str = "none"  # none, cosine, warmup
    first_token_ce_peak: Optional[float] = None
    first_token_ce_warmup_frac: float = 0.4
    first_token_autoscale: str = "yes"
    first_token_entropy_weight: float = 0.0

    # K-token supervision
    K: int = 4
    k_ce_weight: float = 0.5
    adaptive_k_start: Optional[int] = None
    adaptive_k_end: Optional[int] = None

    # Knowledge distillation
    kd_first_k_weight: float = 1.0
    kd_tau: float = 2.0
    kd_skip_text: bool = False

    # Hidden-state KD
    state_kd_weight: float = 0.0
    state_kd_layers: str = "0,1,2"

    # Alignment losses
    latent_align_weight: float = 0.0
    latent_prefix_align_weight: float = 0.0
    latent_align_metric: str = "cosine"
    manifold_stat_weight: float = 0.0


@dataclass
class CurriculumConfig:
    """Training curriculum and warm-up schedules."""
    # Latent dropout curriculum
    latent_keep_start: float = 1.0
    latent_keep_end: float = 1.0
    latent_keep_power: float = 1.0

    # Text/latent mixing warm-up
    warmup_text_latent_steps: int = 0
    warmup_text_latent_epochs: float = 0.0
    warmup_tail_prob: float = 0.0

    # Warm-up loss weights
    warmup_align_tokens: int = 1
    warmup_align_weight: float = 1.0
    warmup_text_teacher_weight: float = 1.0
    warmup_text_latent_weight: float = 0.2
    warmup_text_latent_weight_end: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer and training dynamics."""
    lr: float = 1e-4
    max_grad_norm: float = 1.0
    grad_ckpt: bool = False


@dataclass
class AnchorConfig:
    """Anchor token configuration for latent conditioning."""
    warm_anchor_text: str = ""
    warm_anchor_mode: str = "auto"  # auto, text, chat, none
    train_append_bos_after_prefix: str = "no"  # auto, yes, no
    max_anchor_tokens: int = 32
    use_chat_template: bool = False


@dataclass
class DiagnosticsConfig:
    """Logging and diagnostic configuration."""
    grad_diag_interval: int = 0
    grad_diag_components: str = "tf,first,kce,kd,align,latent_align,latent_prefix_align"
    diagnostic_log: str = ""
    debug: bool = False


@dataclass
class CheckpointConfig:
    """Checkpointing and resumption."""
    save_dir: str = "./ckpt"
    save_every: int = 0
    resume_from: str = ""
    auto_resume: bool = False
    no_load_optimizer: bool = False
    no_load_lr_scheduler: bool = False
    reset_epoch: bool = False
    save_training_stats: bool = False


@dataclass
class SystemConfig:
    """System-level configuration."""
    require_cuda: str = "yes"
    fp16_mps: bool = False


@dataclass
class TrainingConfig:
    """Complete training configuration - top-level container."""

    # Core configuration groups
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    # Feature toggles (all default False)
    features: FeatureToggles = field(default_factory=FeatureToggles)

    # Feature-specific configs (only used if corresponding toggle is True)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    prefix: PrefixTuningConfig = field(default_factory=PrefixTuningConfig)
    deep_prefix: DeepPrefixConfig = field(default_factory=DeepPrefixConfig)
    latent_adapters: LatentAdapterConfig = field(default_factory=LatentAdapterConfig)
    coprocessor: CoprocessorConfig = field(default_factory=CoprocessorConfig)
    gist: GistConfig = field(default_factory=GistConfig)
    latent_refiner: LatentRefinerConfig = field(default_factory=LatentRefinerConfig)

    # Training dynamics
    losses: LossWeights = field(default_factory=LossWeights)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)

    # Infrastructure
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # Milestone 0: Baseline verification mode
    baseline_verification: bool = False

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors.

        Returns:
            List of validation messages. Empty list means valid config.
        """
        warnings = []

        # Validate encoder trainability
        if self.encoder.freeze_encoder and self.encoder.train_encoder:
            warnings.append(
                "WARNING: Both freeze_encoder=True and train_encoder=True set. "
                "freeze_encoder takes precedence."
            )
            self.encoder.train_encoder = False

        # Validate feature combinations
        if self.features.use_deep_prefix and self.features.use_latent_adapters:
            warnings.append(
                "WARNING: Both deep_prefix and latent_adapters enabled. "
                "Research papers typically use ONE approach. May cause interference."
            )

        if self.features.use_deep_prefix and self.features.use_coprocessor:
            warnings.append(
                "ERROR: deep_prefix and coprocessor are mutually exclusive. Disable one of the features."
            )

        # Validate dataset choice
        if self.data.dataset not in ["hotpot", "squad", "squad_v2"]:
            warnings.append(f"ERROR: Unknown dataset '{self.data.dataset}'")

        # Validate encoder type
        if self.encoder.encoder_type not in ["byte", "simple-st", "stq"]:
            warnings.append(f"ERROR: Unknown encoder_type '{self.encoder.encoder_type}'")

        # Validate latent lengths
        if self.encoder.latent_len <= 0:
            warnings.append("ERROR: latent_len must be positive")

        if self.encoder.latent_private_len < 0:
            warnings.append("ERROR: latent_private_len cannot be negative")

        # Validate feature-specific configs only if features are enabled
        if not self.features.use_lora and any([
            self.lora.lora_r != 8,
            self.lora.lora_alpha != 16,
        ]):
            warnings.append(
                "INFO: LoRA config specified but use_lora=False. "
                "Config will be ignored unless feature is enabled."
            )

        if self.features.use_deep_prefix and self.deep_prefix.deep_prefix_len is not None:
            if self.deep_prefix.deep_prefix_len <= 0:
                warnings.append("ERROR: deep_prefix_len must be positive when set")

        # Validate baseline verification mode
        if self.baseline_verification:
            # Check that advanced features will be disabled
            if any([
                self.features.use_deep_prefix,
                self.features.use_latent_adapters,
                self.features.use_gist_head,
            ]):
                warnings.append(
                    "INFO: baseline_verification=True will override feature toggles "
                    "(deep_prefix, latent_adapters, gist_head will be disabled)"
                )

        return warnings

    def apply_baseline_verification(self):
        """Apply Milestone 0 baseline verification overrides."""
        if not self.baseline_verification:
            return

        # Disable advanced features
        self.features.use_deep_prefix = False
        self.features.use_latent_adapters = False
        self.features.use_coprocessor = False
        self.features.use_gist_head = False

        # Zero out KD weights
        self.losses.kd_first_k_weight = 0.0
        self.losses.state_kd_weight = 0.0
        self.gist.gist_weight = 0.0

        # Ensure encoder stays trainable
        self.encoder.train_encoder = True
        self.encoder.freeze_encoder = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    def to_json(self, path: str):
        """Save config to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Load config from dictionary."""
        # Recursively construct nested dataclasses
        return cls(
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            encoder=EncoderConfig(**data.get("encoder", {})),
            adapter=AdapterConfig(**data.get("adapter", {})),
            features=FeatureToggles(**data.get("features", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            prefix=PrefixTuningConfig(**data.get("prefix", {})),
            deep_prefix=DeepPrefixConfig(**data.get("deep_prefix", {})),
            latent_adapters=LatentAdapterConfig(**data.get("latent_adapters", {})),
            coprocessor=CoprocessorConfig(**data.get("coprocessor", {})),
            gist=GistConfig(**data.get("gist", {})),
            latent_refiner=LatentRefinerConfig(**data.get("latent_refiner", {})),
            losses=LossWeights(**data.get("losses", {})),
            curriculum=CurriculumConfig(**data.get("curriculum", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            anchor=AnchorConfig(**data.get("anchor", {})),
            diagnostics=DiagnosticsConfig(**data.get("diagnostics", {})),
            checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
            system=SystemConfig(**data.get("system", {})),
            baseline_verification=data.get("baseline_verification", False),
        )

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        """Construct config from argparse Namespace.

        Args:
            args: argparse.Namespace from train.py

        Returns:
            TrainingConfig instance
        """
        # Helper to safely get attribute with default
        def get(name, default=None):
            return getattr(args, name, default)

        config = cls(
            model=ModelConfig(
                llama_id=get("llama_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                qwen_id=get("qwen_id", "Qwen/Qwen2-0.5B-Instruct"),
                llama_device_map=get("llama_device_map"),
                qwen_device_map=get("qwen_device_map"),
                llama_devices=get("llama_devices"),
                qwen_devices=get("qwen_devices"),
                models=get("models", "llama,qwen"),
                load_4bit=get("load_4bit", False),
                sequential_models=get("sequential_models", False),
                gpu_mem_gib=get("gpu_mem_gib", 78.0),
                teacher_llama_id=get("teacher_llama_id"),
                teacher_qwen_id=get("teacher_qwen_id"),
            ),
            data=DataConfig(
                dataset=get("dataset", "hotpot"),
                hotpot_config=get("hotpot_config", "fullwiki"),
                samples=get("samples", 128),
                epochs=get("epochs", 1),
                batch_size=get("batch_size", 1),
                grad_accum_steps=get("grad_accum_steps", 1),
                max_answer_tokens=get("max_answer_tokens", 32),
                seed=get("seed", 42),
                data_seed=get("data_seed", 42),
            ),
            encoder=EncoderConfig(
                encoder_type=get("encoder_type", "byte"),
                latent_len=get("latent_len", 8),
                latent_shared_len=get("latent_shared_len"),
                latent_private_len=get("latent_private_len", 0),
                d_z=get("d_z", 256),
                max_bytes=get("max_bytes", 512),
                encoder_use_chat_template=get("encoder_use_chat_template", False),
                encoder_backbone=get("encoder_backbone"),
                hf_encoder_id=get("hf_encoder_id", "sentence-transformers/all-MiniLM-L6-v2"),
                max_enc_tokens=get("max_enc_tokens", 1024),
                train_encoder=not get("freeze_encoder", False),  # train_encoder = !freeze_encoder
                freeze_encoder=get("freeze_encoder", False),
            ),
            adapter=AdapterConfig(
                adapter_hidden_mult=get("adapter_hidden_mult", 2),
                adapter_dropout=get("adapter_dropout", 0.0),
                adapter_colorize=get("adapter_colorize", False),
                adapter_metadata=get("adapter_metadata", True),
                adapter_freeze_scale=get("adapter_freeze_scale", False),
                scale_l2=get("scale_l2", 0.05),
                adapter_rms_l2=get("adapter_rms_l2", 0.0),
            ),
            features=FeatureToggles(
                use_lora=get("use_lora", False),
                use_prefix=get("use_prefix", False),
                use_deep_prefix=get("use_deep_prefix", False),
                use_latent_adapters=get("use_latent_adapters", False),
                 use_coprocessor=get("use_coprocessor", False),
                use_gist_head=get("use_gist_head", False),
                use_latent_refiner=(get("latent_refiner_layers", 0) or 0) > 0,
            ),
            lora=LoRAConfig(
                lora_r=get("lora_r", 8),
                lora_alpha=get("lora_alpha", 16),
                lora_dropout=get("lora_dropout", 0.05),
                lora_firstN=get("lora_firstN"),
                lora_target_modules=get("lora_target_modules", "auto"),
            ),
            prefix=PrefixTuningConfig(
                prefix_tokens=get("prefix_tokens", 16),
                prefix_projection=get("prefix_projection", False),
                peft_prefix_all_layers=get("peft_prefix_all_layers", "yes"),
            ),
            deep_prefix=DeepPrefixConfig(
                deep_prefix_len=get("deep_prefix_len"),
                deep_prefix_dropout=get("deep_prefix_dropout", 0.1),
            ),
            latent_adapters=LatentAdapterConfig(
                latent_adapter_layers=get("latent_adapter_layers", "8,16,24"),
                latent_adapter_heads=get("latent_adapter_heads", 8),
                latent_adapter_dropout=get("latent_adapter_dropout", 0.1),
            ),
            coprocessor=CoprocessorConfig(
                coprocessor_len=get("coprocessor_len", 1),
                coprocessor_width=get("coprocessor_width", 256),
                coprocessor_dropout=get("coprocessor_dropout", 0.1),
                coprocessor_kv_scale=get("coprocessor_kv_scale", 0.8),
                coprocessor_pool=get("coprocessor_pool", "mean"),
                coprocessor_heads=get("coprocessor_heads", ""),
            ),
            gist=GistConfig(
                gist_target_len=get("gist_target_len", 48),
                gist_hidden=get("gist_hidden", 512),
                gist_layers=get("gist_layers", 2),
                gist_dropout=get("gist_dropout", 0.1),
                gist_weight=get("gist_weight", 0.0),
                gist_mask_prob=get("gist_mask_prob", 0.15),
            ),
            latent_refiner=LatentRefinerConfig(
                latent_refiner_layers=get("latent_refiner_layers", 0),
                latent_refiner_heads=get("latent_refiner_heads", 4),
            ),
            losses=LossWeights(
                first_token_ce_weight=get("first_token_ce_weight", 0.5),
                first_token_ce_schedule=get("first_token_ce_schedule", "none"),
                first_token_ce_peak=get("first_token_ce_peak"),
                first_token_ce_warmup_frac=get("first_token_ce_warmup_frac", 0.4),
                first_token_autoscale=get("first_token_autoscale", "yes"),
                first_token_entropy_weight=get("first_token_entropy_weight", 0.0),
                K=get("K", 4),
                k_ce_weight=get("k_ce_weight", 0.5),
                adaptive_k_start=get("adaptive_k_start"),
                adaptive_k_end=get("adaptive_k_end"),
                kd_first_k_weight=get("kd_first_k_weight", 1.0),
                kd_tau=get("kd_tau", 2.0),
                kd_skip_text=get("kd_skip_text", False),
                state_kd_weight=get("state_kd_weight", 0.0),
                state_kd_layers=get("state_kd_layers", "0,1,2"),
                latent_align_weight=get("latent_align_weight", 0.0),
                latent_prefix_align_weight=get("latent_prefix_align_weight", 0.0),
                latent_align_metric=get("latent_align_metric", "cosine"),
                manifold_stat_weight=get("manifold_stat_weight", 0.0),
            ),
            curriculum=CurriculumConfig(
                latent_keep_start=get("latent_keep_start", 1.0),
                latent_keep_end=get("latent_keep_end", 1.0),
                latent_keep_power=get("latent_keep_power", 1.0),
                warmup_text_latent_steps=get("warmup_text_latent_steps", 0),
                warmup_text_latent_epochs=get("warmup_text_latent_epochs", 0.0),
                warmup_tail_prob=get("warmup_tail_prob", 0.0),
                warmup_align_tokens=get("warmup_align_tokens", 1),
                warmup_align_weight=get("warmup_align_weight", 1.0),
                warmup_text_teacher_weight=get("warmup_text_teacher_weight", 1.0),
                warmup_text_latent_weight=get("warmup_text_latent_weight", 0.2),
                warmup_text_latent_weight_end=get("warmup_text_latent_weight_end", 1.0),
            ),
            optimizer=OptimizerConfig(
                lr=get("lr", 1e-4),
                max_grad_norm=get("max_grad_norm", 1.0),
                grad_ckpt=get("grad_ckpt", False),
            ),
            anchor=AnchorConfig(
                warm_anchor_text=get("warm_anchor_text", ""),
                warm_anchor_mode=get("warm_anchor_mode", "auto"),
                train_append_bos_after_prefix=get("train_append_bos_after_prefix", "no"),
                max_anchor_tokens=get("max_anchor_tokens", 32),
                use_chat_template=get("use_chat_template", False),
            ),
            diagnostics=DiagnosticsConfig(
                grad_diag_interval=get("grad_diag_interval", 0),
                grad_diag_components=get("grad_diag_components", "tf,first,kce,kd,align,latent_align,latent_prefix_align"),
                diagnostic_log=get("diagnostic_log", ""),
                debug=get("debug", False),
            ),
            checkpoint=CheckpointConfig(
                save_dir=get("save_dir", "./ckpt"),
                save_every=get("save_every", 0),
                resume_from=get("resume_from", ""),
                auto_resume=get("auto_resume", False),
                no_load_optimizer=get("no_load_optimizer", False),
                no_load_lr_scheduler=get("no_load_lr_scheduler", False),
                reset_epoch=get("reset_epoch", False),
                save_training_stats=get("save_training_stats", False),
            ),
            system=SystemConfig(
                require_cuda=get("require_cuda", "yes"),
                fp16_mps=get("fp16_mps", False),
            ),
            baseline_verification=get("baseline_verification", False),
        )

        # Apply baseline verification if requested
        if config.baseline_verification:
            config.apply_baseline_verification()

        return config
