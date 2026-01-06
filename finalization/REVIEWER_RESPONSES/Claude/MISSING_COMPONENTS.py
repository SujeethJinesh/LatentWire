# =============================================================================
# MISSING COMPONENTS TO ADD TO LATENTWIRE.py
# =============================================================================
# Add these after line ~1210 (after the Evaluator class) and before the CLI

# ============================================================================
# LINEAR PROBE BASELINE
# ============================================================================

class LinearProbeBaseline:
    """Linear probe baseline using sklearn's LogisticRegression.
    
    This baseline extracts hidden states from a frozen LLM and trains
    a linear classifier on top. Used to verify that the Bridge architecture
    provides value beyond simple linear projection.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        layer_idx: int = 16,  # Middle layer
        pooling: str = "mean",
        max_samples: int = 5000,
        device: str = None,
    ):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.max_samples = max_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.probe = None
        self.scaler = None
        
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
    
    def _load_model(self):
        """Lazy load model."""
        if self.model is None and HAS_TRANSFORMERS:
            print(f"Loading model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                output_hidden_states=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
    
    def extract_features(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Extract hidden state features from texts."""
        self._load_model()
        
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[self.layer_idx]  # [B, L, D]
            
            # Pool across sequence dimension
            if self.pooling == "mean":
                # Mask padding
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            elif self.pooling == "last":
                # Get last non-padded token
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                pooled = hidden_states[torch.arange(len(batch_texts)), seq_lens]
            elif self.pooling == "first":
                pooled = hidden_states[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
            
            all_features.append(pooled.cpu().numpy())
        
        return np.vstack(all_features)
    
    def train(
        self,
        texts: List[str],
        labels: List[int],
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train linear probe with cross-validation."""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for linear probe")
        
        # Limit samples
        if self.max_samples > 0 and len(texts) > self.max_samples:
            indices = np.random.choice(len(texts), self.max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        print(f"Extracting features from {len(texts)} samples...")
        X = self.extract_features(texts)
        y = np.array(labels)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train with cross-validation
        print("Training linear probe...")
        self.probe = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial" if len(np.unique(y)) > 2 else "ovr",
            solver="lbfgs",
            n_jobs=-1,
        )
        
        # Cross-validation scores
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        cv_scores = cross_val_score(
            self.probe,
            X, y,
            cv=StratifiedKFold(n_splits=cv_folds),
            scoring="accuracy",
        )
        
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Fit on all data
        self.probe.fit(X, y)
        train_acc = (self.probe.predict(X) == y).mean()
        
        return {
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "train_accuracy": float(train_acc),
        }
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Evaluate trained probe on test data."""
        if self.probe is None:
            raise ValueError("Probe not trained!")
        
        X = self.extract_features(texts)
        X = self.scaler.transform(X)
        y = np.array(labels)
        
        predictions = self.probe.predict(X)
        probs = self.probe.predict_proba(X)
        
        from sklearn.metrics import accuracy_score, f1_score
        
        return {
            "accuracy": float(accuracy_score(y, predictions)),
            "f1_macro": float(f1_score(y, predictions, average="macro")),
            "f1_weighted": float(f1_score(y, predictions, average="weighted")),
        }


# ============================================================================
# STATISTICAL TESTING
# ============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, Tuple[float, float]]:
    """Compute bootstrap confidence interval.
    
    Args:
        data: 1D array of observations
        statistic: Function to compute statistic (default: mean)
        n_resamples: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed
        
    Returns:
        (point_estimate, (lower_bound, upper_bound))
    """
    data = np.asarray(data)
    n = len(data)
    
    if n < 2:
        return float(statistic(data)), (float(data[0]), float(data[0]))
    
    rng = np.random.default_rng(random_state)
    
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_resamples):
        resample = data[rng.choice(n, size=n, replace=True)]
        bootstrap_stats.append(statistic(resample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentile CI
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    point_estimate = statistic(data)
    
    return float(point_estimate), (float(lower), float(upper))


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10000,
    alternative: str = "two-sided",
    random_state: Optional[int] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Paired bootstrap test for comparing two methods.
    
    Args:
        scores_a: Scores from method A
        scores_b: Scores from method B  
        n_resamples: Number of bootstrap resamples
        alternative: 'two-sided', 'greater', or 'less'
        random_state: Random seed
        
    Returns:
        (observed_diff, p_value, stats_dict)
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")
    
    n = len(scores_a)
    diffs = scores_a - scores_b
    observed_diff = np.mean(diffs)
    
    # Bootstrap under null (centered at 0)
    centered_diffs = diffs - observed_diff
    
    rng = np.random.default_rng(random_state)
    bootstrap_means = []
    
    for _ in range(n_resamples):
        resample = centered_diffs[rng.choice(n, size=n, replace=True)]
        bootstrap_means.append(np.mean(resample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(bootstrap_means) >= np.abs(observed_diff))
    elif alternative == "greater":
        p_value = np.mean(bootstrap_means >= observed_diff)
    elif alternative == "less":
        p_value = np.mean(bootstrap_means <= observed_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    # Minimum p-value
    p_value = max(p_value, 1.0 / n_resamples)
    
    stats = {
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "std_a": float(np.std(scores_a, ddof=1)),
        "std_b": float(np.std(scores_b, ddof=1)),
        "n": n,
    }
    
    return float(observed_diff), float(p_value), stats


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """Paired t-test for comparing two methods.
    
    Returns:
        (t_statistic, p_value)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for t-test")
    
    from scipy.stats import ttest_rel
    
    result = ttest_rel(scores_a, scores_b)
    return float(result.statistic), float(result.pvalue)


# ============================================================================
# DATA HELPERS FOR BASELINES
# ============================================================================

def get_dataset_for_probe(
    dataset_name: str,
    max_train: int = 5000,
    max_test: int = 500,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load dataset in format suitable for linear probe baseline.
    
    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required")
    
    rng = np.random.default_rng(seed)
    
    if dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")
        train_data = dataset["train"]
        test_data = dataset["validation"]
        
        text_key = "sentence"
        label_key = "label"
        
    elif dataset_name == "agnews":
        dataset = load_dataset("ag_news")
        train_data = dataset["train"]
        test_data = dataset["test"]
        
        text_key = "text"
        label_key = "label"
        
    elif dataset_name == "trec":
        dataset = load_dataset("trec")
        train_data = dataset["train"]
        test_data = dataset["test"]
        
        text_key = "text"
        label_key = "coarse_label"
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Sample training data
    train_indices = rng.choice(len(train_data), min(max_train, len(train_data)), replace=False)
    train_texts = [train_data[int(i)][text_key] for i in train_indices]
    train_labels = [train_data[int(i)][label_key] for i in train_indices]
    
    # Sample test data
    test_indices = rng.choice(len(test_data), min(max_test, len(test_data)), replace=False)
    test_texts = [test_data[int(i)][text_key] for i in test_indices]
    test_labels = [test_data[int(i)][label_key] for i in test_indices]
    
    return train_texts, train_labels, test_texts, test_labels


# ============================================================================
# VISUALIZATION (Basic)
# ============================================================================

class Visualizer:
    """Basic visualization utilities."""
    
    def __init__(self, save_dir: str = "figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        metrics: Dict[str, List[float]],
        save_name: str = "training_curves.pdf",
    ):
        """Plot training loss and accuracy curves."""
        if not HAS_PLOTTING:
            print("Plotting not available (matplotlib not installed)")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        if "loss" in metrics:
            axes[0].plot(metrics["loss"], label="Loss")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss")
            axes[0].legend()
        
        # Accuracy curve
        if "accuracy" in metrics:
            axes[1].plot(metrics["accuracy"], label="Accuracy")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("First Token Accuracy")
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved: {self.save_dir / save_name}")
    
    def plot_comparison_bars(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "accuracy",
        save_name: str = "comparison.pdf",
    ):
        """Plot bar chart comparing methods."""
        if not HAS_PLOTTING:
            print("Plotting not available")
            return
        
        methods = list(results.keys())
        values = [results[m].get(metric, 0) for m in methods]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(methods, values, color=["gray", "blue", "green", "orange"][:len(methods)])
        
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} by Method")
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{val:.3f}", ha="center", va="bottom")
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved: {self.save_dir / save_name}")
