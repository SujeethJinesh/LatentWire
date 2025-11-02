"""Constants for cross-model experiments."""

# Model identifiers
LLAMA_8B = "meta-llama/Llama-3.1-8B"
LLAMA_3B = "meta-llama/Llama-3.2-3B"
MISTRAL_7B = "mistralai/Mistral-7B-v0.3"

# Ramesh & Li (ICML 2025) configuration
RAMESH_LI_LAYER = 26  # Layer 26 from "Communicating Activations Between LLM Agents"

# Test prompts for text similarity evaluation
TEST_PROMPTS = [
    "The capital of France is",
    "To solve this problem, we need to",
    "The future of artificial intelligence is",
    "In the year 2050,",
    "The main difference between cats and dogs is"
]
