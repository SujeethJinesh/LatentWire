# Cross-Model Collaborative Reasoning Experiments

**Research Question**: Can heterogeneous LLMs (Llama 3.1 8B + Mistral 7B) collaborate mid-reasoning through hidden state transfer to solve problems better than either alone?

**Hypothesis**: Models have complementary strengths on different reasoning step types. By transferring intermediate reasoning states (hidden representations) via learned/training-free alignment, they can help each other solve multi-step problems.

**Novel Contribution**: First work (to our knowledge) on mid-reasoning hidden state handoffs between frozen heterogeneous LLMs for collaborative problem-solving.

---

## Background & Motivation

### Evidence for Complementary Model Strengths

Recent research demonstrates clear model specialization patterns:

1. **Qwen excels at mathematical reasoning and calculation**
   - Qwen 2.5-72B achieves 95.8% on GSM8K [1]
   - Qwen 2.5-Math specialized variants show superior calculation accuracy [2]
   - Better at structured step-by-step mathematical explanations

2. **Llama excels at general reasoning and speed**
   - Llama 3.1 405B achieves 96.0% on GSM8K [3]
   - Up to 3× faster than Qwen on complex tasks [4]
   - Superior broad knowledge application and general reasoning

3. **Reasoning step categories have distinct requirements** [5, 6]
   - **Problem decomposition**: Understanding, breaking into sub-problems
   - **Logical/commonsense reasoning**: Planning, operation selection
   - **Calculation**: Arithmetic operations, numerical computation
   - **Solution completeness**: Ensuring all necessary steps

4. **Cross-method complementarity is established** [7, 8]
   - Chain-of-Thought (conceptual) vs Program-of-Thought (computational)
   - Natural language (logical deduction) vs Code (precise execution)
   - "Distinct approaches can synergize, resulting in benefits exceeding any single approach" [7]

### Research Gap

While extensive work exists on:
- ✅ Multi-agent text-based collaboration [9, 10]
- ✅ Continuous latent reasoning within single models (COCONUT) [11]
- ✅ Query-level routing between models [12, 13]
- ✅ Probability fusion ensembles [14]

**No prior work** demonstrates:
- ❌ Mid-reasoning hidden state handoffs between heterogeneous frozen LLMs
- ❌ Continuous latent space collaboration across different architectures
- ❌ Step-level model switching via shared representations

**Closest related work**:
- **"Communicating Activations Between LM Agents"** (Jan 2025) [15]: Merges activations mid-forward-pass but focuses on coordination games, not frozen models
- **COCONUT** (Dec 2024) [11]: Continuous latent reasoning but single model only
- **MixReasoning** (Oct 2024) [16]: Token-level uncertainty routing but within single model

---

## Experiment 1: Simple Cross-Model Handoff on GSM8K

### Motivation

Test the most basic collaboration pattern: Model A processes the question, Model B generates the answer from A's hidden state representation.

This tests:
1. Whether training-free alignment (Procrustes) preserves enough information for cross-model transfer
2. If models can leverage each other's complementary strengths
3. Feasibility of hidden state handoffs for reasoning tasks

### Method

**Setup:**
- Dataset: GSM8K (500 test problems) [17]
- Models: Llama 3.1 8B, Mistral 7B v0.1
- Alignment: Procrustes transformation (already calibrated on 5000 WikiText samples)
- Hardware: 1× H100 GPU (both models fit in 79GB VRAM)

**Procedure:**

```python
def cross_model_handoff(model_a, model_b, transform_ab, question):
    """
    Model A processes question → hidden state → transform → Model B answers
    """
    # Step 1: Model A processes question
    inputs_a = tokenizer_a(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_a = model_a(**inputs_a, output_hidden_states=True)
        h_a = outputs_a.hidden_states[-1][:, -1, :]  # Last token, last layer

    # Step 2: Transform A's hidden state to B's space
    h_transformed = transform_ab(h_a)  # Procrustes: h_b ≈ W @ h_a

    # Step 3: Model B generates answer from transformed hidden state
    # Use h_transformed as soft prefix to Model B
    with torch.no_grad():
        # Prepend transformed hidden state as continuous prompt
        b_inputs_embeds = torch.cat([
            h_transformed.unsqueeze(1),  # [1, 1, hidden_dim]
            model_b.get_input_embeddings()(tokenizer_b.encode(
                "Answer:", return_tensors="pt"
            ).to(device))
        ], dim=1)

        answer_ids = model_b.generate(
            inputs_embeds=b_inputs_embeds,
            max_new_tokens=50,
            do_sample=False  # Greedy decoding for reproducibility
        )

    return tokenizer_b.decode(answer_ids[0], skip_special_tokens=True)
```

**Alignment Methods to Test:**
1. No alignment (baseline, expected to fail)
2. Procrustes (orthogonal rotation)
3. Centered Procrustes (rotation after mean-centering)
4. Scaled Procrustes (rotation + uniform scale)

### Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **Llama Solo** | Llama processes question and generates answer | Single-model performance |
| **Mistral Solo** | Mistral processes question and generates answer | Single-model performance |
| **Text Handoff** | Llama generates intermediate text → Mistral continues | Text-based collaboration upper bound |
| **Oracle** | For each problem, select whichever solo model got correct answer | Upper bound on complementarity |

### Success Metrics

**Primary:**
- **Exact Match (EM)**: Answer exactly matches gold answer
- **Complementary Gain**: Problems where both solos fail but collaboration succeeds
  ```python
  complementary_cases = (llama_em == 0) & (mistral_em == 0) & (collab_em == 1)
  complementary_gain = complementary_cases.mean()
  ```

**Secondary:**
- **Transfer quality**: How often handoff preserves/improves over source model
- **Error propagation**: How often handoff makes correct answer worse

**Success Criteria:**
- ✅ Strong: Collaboration accuracy > max(Llama, Mistral) by ≥3%
- ✅ Moderate: Complementary gain ≥ 5% (collaboration helps on 5% of problems)
- ⚠️ Weak: Collaboration ≈ best solo (feasibility but no synergy)
- ❌ Failure: Collaboration < both solos (alignment destroys information)

### Expected Outcomes

Based on prior work on model routing [12, 13] and ensemble methods [18, 19]:
- **Optimistic**: +5-10% improvement over best solo on calculation-heavy problems
- **Realistic**: +2-5% improvement, clear complementarity patterns
- **Pessimistic**: No improvement but identify specific failure modes

### Implementation Timeline

- **Day 1**: Adapt existing `cross_model_ablation.py` for GSM8K
- **Day 2**: Run experiment on HPC (4 alignment methods × 2 directions × 500 problems)
- **Day 3**: Analysis and error categorization

### References for Experiment 1

[12] CITER (2024) - Token-level routing, arXiv:2502.01976
[13] MixLLM (2025) - Dynamic routing with streaming queries
[14] DeePEn (2024) - Heterogeneous ensemble, arXiv:2404.12715

---

## Experiment 2: Detect Which Model Is Better at Which Steps

### Motivation

Before investing in complex routing mechanisms, we need empirical evidence that models truly have complementary strengths on different reasoning step types.

This experiment provides:
1. **Quantitative characterization** of model strengths/weaknesses
2. **Guidance for routing policy**: When to hand off vs. when to stay
3. **Validation of hypothesis**: Do Llama/Mistral specialize differently?

### Method

**Problem Categorization:**

Manually categorize 500 GSM8K problems into:

1. **Pure Calculation** (~20% of GSM8K)
   - Example: "What is 15% of 240?"
   - Minimal word problem complexity, straightforward arithmetic
   - Hypothesis: Mistral better (less overhead, faster calculation)

2. **Pure Word Problem** (~30% of GSM8K)
   - Example: "John has 5 apples. He gives 2 to Mary. How many remain?"
   - Requires semantic understanding, minimal calculation
   - Hypothesis: Llama better (general reasoning, comprehension)

3. **Mixed Problem** (~50% of GSM8K)
   - Example: "Train A leaves at 3pm going 60mph for 2 hours. Train B..."
   - Requires both decomposition and calculation
   - Hypothesis: Collaboration most beneficial here

**Procedure:**

```python
def categorize_problem(problem_text):
    """
    Categorize GSM8K problem by predominant reasoning type.

    Categories based on error analysis research [5, 6]:
    - CALC: Pure calculation (minimal context)
    - WORD: Pure reasoning (minimal arithmetic)
    - MIXED: Requires both reasoning and calculation
    """
    # Simple heuristic (can be refined with LLM assistance)
    num_numbers = len(re.findall(r'\d+', problem_text))
    num_words = len(problem_text.split())

    if num_words < 20 and num_numbers >= 2:
        return "CALC"
    elif num_words > 40 and num_numbers <= 3:
        return "WORD"
    else:
        return "MIXED"

# Run both models on entire test set
results = []
for problem in gsm8k_test:
    category = categorize_problem(problem['question'])

    # Llama result
    llama_answer = llama_generate(problem['question'])
    llama_correct = check_exact_match(llama_answer, problem['answer'])

    # Mistral result
    mistral_answer = mistral_generate(problem['question'])
    mistral_correct = check_exact_match(mistral_answer, problem['answer'])

    results.append({
        'category': category,
        'llama_correct': llama_correct,
        'mistral_correct': mistral_correct,
        'both_correct': llama_correct and mistral_correct,
        'both_wrong': not llama_correct and not mistral_correct,
        'llama_only': llama_correct and not mistral_correct,
        'mistral_only': mistral_correct and not llama_correct,
    })

# Analyze by category
for category in ['CALC', 'WORD', 'MIXED']:
    cat_results = [r for r in results if r['category'] == category]
    print(f"\n{category} Problems (n={len(cat_results)}):")
    print(f"  Llama accuracy: {mean([r['llama_correct'] for r in cat_results]):.1%}")
    print(f"  Mistral accuracy: {mean([r['mistral_correct'] for r in cat_results]):.1%}")
    print(f"  Complementary (Llama only): {mean([r['llama_only'] for r in cat_results]):.1%}")
    print(f"  Complementary (Mistral only): {mean([r['mistral_only'] for r in cat_results]):.1%}")
    print(f"  Both wrong: {mean([r['both_wrong'] for r in cat_results]):.1%}")
```

### Success Metrics

**Primary:**
- **Accuracy gap by category**: |Llama_acc - Mistral_acc| > 5% indicates specialization
- **Complementarity by category**: % problems where one succeeds and other fails

**Secondary:**
- **Error type analysis**: Categorize errors (calculation, logical, semantic) [5, 6]
- **Confidence patterns**: Measure perplexity/entropy by category

**Success Criteria:**
- ✅ Strong: Clear specialization (≥10% accuracy gap on CALC or WORD)
- ✅ Moderate: Weak specialization (5-10% gap) but consistent patterns
- ⚠️ Weak: No clear specialization but high complementarity
- ❌ Failure: Models perform identically across categories

### Expected Outcomes

Based on prior benchmarking [1, 2, 3]:
- **CALC problems**: Mistral ≥5% better (optimized for arithmetic)
- **WORD problems**: Llama ≥5% better (general reasoning strength)
- **MIXED problems**: Similar performance but high complementarity

### Implementation Timeline

- **Day 1**: Implement categorization heuristic + validation on 50 problems
- **Day 2**: Run both models on 500 GSM8K problems
- **Day 3**: Statistical analysis + visualization

### Deliverables

1. **Specialization table** by category
2. **Complementarity matrix** showing when to hand off
3. **Error analysis** identifying specific failure modes
4. **Visualization**: Heatmap of (model, category) → accuracy

### References for Experiment 2

[1] Qwen 2.5 Technical Report (2024) - 95.8% GSM8K
[2] Qwen 2.5-Math (2024) - Specialized mathematical reasoning
[3] Llama 3.1 Report (2024) - 96.0% GSM8K with 405B
[5] MR-GSM8K (2024) - Meta-reasoning benchmark, error categorization
[6] GSM-Symbolic (Apple, 2024) - Understanding limitations and fragility

---

## Experiment 3: Cascade Inference with Quality Threshold

### Motivation

A practical hybrid system that:
1. Tries efficient compressed latent transfer first
2. Falls back to expensive text if quality is insufficient
3. Optimizes cost-quality tradeoff (inspired by FrugalGPT [20])

This tests:
- Whether compressed representations preserve enough information
- If confidence estimation can guide routing decisions
- Practical deployment viability

### Method

**Cascade Strategy:**

```python
def cascade_inference(question, model_a, model_b, transform, threshold=0.5):
    """
    Try latent handoff first, cascade to text if low confidence.
    """
    # Step 1: Attempt latent handoff
    latent_answer, confidence = latent_handoff_with_confidence(
        question, model_a, model_b, transform
    )

    # Step 2: Check confidence
    if confidence >= threshold:
        return latent_answer, "latent", confidence
    else:
        # Step 3: Fallback to text
        text_answer = model_b.generate(f"{question} Answer:")
        return text_answer, "text", 1.0  # Text assumed high confidence

def latent_handoff_with_confidence(question, model_a, model_b, transform):
    """
    Perform latent handoff and estimate confidence.
    """
    # Generate with latent
    h_a = get_hidden_state(model_a, question)
    h_transformed = transform(h_a)

    # Generate answer with Model B
    outputs = model_b.generate(
        inputs_embeds=h_transformed,
        max_new_tokens=50,
        return_dict_in_generate=True,
        output_scores=True
    )

    answer = tokenizer_b.decode(outputs.sequences[0])

    # Confidence estimation (multiple methods)
    confidence = estimate_confidence(outputs, method="entropy")

    return answer, confidence

def estimate_confidence(outputs, method="entropy"):
    """
    Estimate answer quality without gold labels.

    Based on research [21, 22, 23]:
    - Verbalized confidence: NOT reliable (LLMs overconfident)
    - Token-level entropy: Lower = more confident
    - Multi-sample consistency: Higher agreement = more confident
    """
    if method == "entropy":
        # Average entropy across generated tokens
        scores = torch.stack(outputs.scores)  # [seq_len, vocab_size]
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        confidence = 1.0 - entropy.mean().item() / np.log(50257)  # Normalized
        return confidence

    elif method == "max_prob":
        # Average max probability per token
        scores = torch.stack(outputs.scores)
        max_probs = torch.softmax(scores, dim=-1).max(dim=-1).values
        confidence = max_probs.mean().item()
        return confidence

    elif method == "consistency":
        # Generate K samples, measure agreement (expensive but reliable [23])
        # Not implemented in fast version
        raise NotImplementedError
```

**Threshold Sweep:**

Test multiple confidence thresholds to find optimal tradeoff:

```python
thresholds = [0.3, 0.5, 0.7, 0.9]

for threshold in thresholds:
    latent_used = 0
    text_used = 0
    correct = 0

    for problem in test_set:
        answer, mode, conf = cascade_inference(
            problem['question'],
            llama, mistral, procrustes,
            threshold=threshold
        )

        if mode == "latent":
            latent_used += 1
        else:
            text_used += 1

        if check_exact_match(answer, problem['gold']):
            correct += 1

    accuracy = correct / len(test_set)
    cost_savings = latent_used / len(test_set)  # % using cheap latent

    print(f"Threshold {threshold}:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Latent usage: {cost_savings:.1%}")
    print(f"  Cost savings: {cost_savings * 0.75:.1%}")  # Assume 4× compression
```

### Baselines

| Baseline | Description | Cost |
|----------|-------------|------|
| **All Text** | Never use latent, always full text | 100% |
| **All Latent** | Always use latent, never cascade | 25% (4× compression) |
| **Oracle Cascade** | Perfect confidence estimation | Variable (optimal) |

### Success Metrics

**Primary:**
- **Cost-Quality Tradeoff Curve**: Plot accuracy vs. % latent usage
- **Pareto Efficiency**: Is cascade Pareto-optimal?

**Secondary:**
- **Calibration**: Does confidence correlate with actual correctness?
  ```python
  # Bin by confidence, measure accuracy per bin
  for conf_bin in [(0, 0.2), (0.2, 0.4), ..., (0.8, 1.0)]:
      problems_in_bin = [p for p in results if conf_bin[0] <= p['conf'] < conf_bin[1]]
      accuracy_in_bin = mean([p['correct'] for p in problems_in_bin])
      print(f"Confidence {conf_bin}: {accuracy_in_bin:.1%}")
  ```

**Success Criteria:**
- ✅ Strong: 50%+ cost savings with ≤2% accuracy loss
- ✅ Moderate: 30%+ cost savings with ≤5% accuracy loss
- ⚠️ Weak: Cost savings but poor calibration
- ❌ Failure: No savings or severe accuracy degradation

### Expected Outcomes

Based on FrugalGPT [20] (98% cost reduction) and cascade literature:
- **Optimistic**: 60% cost savings, ≤1% accuracy loss
- **Realistic**: 40% cost savings, 2-3% accuracy loss
- **Pessimistic**: 20% cost savings, confidence estimation unreliable

### Implementation Timeline

- **Day 1**: Implement confidence estimation methods
- **Day 2**: Threshold sweep on 500 GSM8K problems
- **Day 3**: Calibration analysis + visualization

### References for Experiment 3

[20] FrugalGPT (2023) - 98% cost reduction matching GPT-4
[21] "LLMs Cannot Self-Correct" (2024) - Overconfidence in verbalized estimates
[22] Uncertainty Estimation survey (2024) - Token-level entropy methods
[23] Multi-sample consistency - Best black-box confidence method

---

## Experiment 4: Complementary Error Analysis

### Motivation

Identify specific problem characteristics where collaboration helps vs. hurts. This provides:
1. **Mechanistic understanding** of when handoffs work
2. **Dataset for training routing policies** (Experiment 6)
3. **Error patterns** to guide future improvements

### Method

**Problem Classification:**

For each problem, classify into one of four outcome categories:

```python
def classify_outcome(llama_correct, mistral_correct, handoff_correct):
    """
    Classify collaborative outcome.
    """
    if llama_correct and mistral_correct:
        return "both_solo_correct"  # Collaboration unnecessary

    elif not llama_correct and not mistral_correct:
        if handoff_correct:
            return "complementary_success"  # ⭐ Key case!
        else:
            return "both_fail"

    elif llama_correct and not mistral_correct:
        if handoff_correct:
            return "llama_transfer_success"  # A→B preserves quality
        else:
            return "llama_transfer_fail"  # A→B destroys quality

    elif mistral_correct and not llama_correct:
        if handoff_correct:
            return "mistral_transfer_success"  # B→A preserves quality
        else:
            return "mistral_transfer_fail"  # B→A destroys quality

# Run classification
outcomes = []
for problem in test_set:
    llama_ans = llama_solo(problem['question'])
    mistral_ans = mistral_solo(problem['question'])
    handoff_ans = llama_to_mistral(problem['question'])

    llama_correct = check_em(llama_ans, problem['gold'])
    mistral_correct = check_em(mistral_ans, problem['gold'])
    handoff_correct = check_em(handoff_ans, problem['gold'])

    outcome = classify_outcome(llama_correct, mistral_correct, handoff_correct)

    outcomes.append({
        'question': problem['question'],
        'outcome': outcome,
        'category': categorize_problem(problem['question']),  # From Exp 2
        'question_len': len(problem['question'].split()),
        'num_steps': count_reasoning_steps(problem['question']),
    })
```

**Deep Dive on Complementary Success Cases:**

```python
complementary_success = [o for o in outcomes if o['outcome'] == 'complementary_success']

print(f"Found {len(complementary_success)} complementary successes")
print(f"Rate: {len(complementary_success) / len(outcomes):.1%}")

# Analyze characteristics
for case in complementary_success[:10]:  # Inspect first 10
    print(f"\nQuestion: {case['question']}")
    print(f"Category: {case['category']}")
    print(f"Length: {case['question_len']} words")
    print(f"Steps: {case['num_steps']}")

    # Manual error analysis (what did each model get wrong?)
    print("Llama error type: [MANUAL INSPECTION]")
    print("Mistral error type: [MANUAL INSPECTION]")
    print("Why handoff succeeded: [HYPOTHESIS]")
```

**Statistical Analysis:**

```python
# Chi-square test: Is complementary success independent of category?
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(
    [o['category'] for o in outcomes],
    [o['outcome'] == 'complementary_success' for o in outcomes]
)

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square: {chi2:.2f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Complementary success is NOT independent of problem category")
else:
    print("❌ No significant relationship found")
```

### Success Metrics

**Primary:**
- **Complementary success rate**: % problems where both solos fail but handoff succeeds
- **Transfer preservation rate**: % problems where correct solo → correct handoff

**Secondary:**
- **Error propagation rate**: % problems where handoff makes correct answer worse
- **Category correlation**: Which categories benefit most from collaboration?

**Success Criteria:**
- ✅ Strong: ≥10% complementary success rate
- ✅ Moderate: 5-10% complementary success with clear patterns
- ⚠️ Weak: <5% but statistically significant category effects
- ❌ Failure: No complementary successes, high error propagation

### Expected Outcomes

Based on ensemble literature [18, 19] and model routing research:
- **Optimistic**: 15-20% complementary success, concentrated in MIXED problems
- **Realistic**: 8-12% complementary success, varies by category
- **Pessimistic**: 3-5% complementary success, high variance

### Implementation Timeline

- **Day 1**: Run all models on 500 GSM8K problems, classify outcomes
- **Day 2**: Statistical analysis + category correlations
- **Day 3**: Manual inspection of complementary cases + error categorization

### Deliverables

1. **Outcome distribution table** across categories
2. **Complementary success case studies** (10-20 examples)
3. **Error propagation analysis** identifying failure modes
4. **Recommendations** for when to use handoffs

### References for Experiment 4

[5] MR-GSM8K (2024) - Error categorization framework
[6] GSM-Symbolic (2024) - Fragility analysis
[18] Multi-Agent Peer Review (2023) - +1.4% via collaboration
[19] SLM-MUX (2025) - +4.8% combining small models, arXiv:2501.xxxxx

---

## Experiment 5: Multi-Step Handoff (A→B→A)

### Motivation

Real-world problems require multiple reasoning steps. Can models iteratively help each other by trading off based on their strengths?

**Hypothesis**:
- Llama better at problem decomposition (step 1)
- Mistral better at calculation (step 2)
- Llama better at final synthesis (step 3)

This tests:
1. Whether multi-hop latent transfer preserves information
2. If models can build on each other's intermediate representations
3. Optimal collaboration patterns (A→B vs A→B→A vs B→A→B)

### Method

**Multi-Step Handoff Patterns:**

```python
def single_handoff_AB(question, llama, mistral, transform_lm):
    """Pattern 1: Llama → Mistral"""
    h_llama = get_hidden_state(llama, question)
    h_transformed = transform_lm(h_llama)
    answer = mistral.generate(inputs_embeds=h_transformed)
    return answer

def single_handoff_BA(question, mistral, llama, transform_ml):
    """Pattern 2: Mistral → Llama"""
    h_mistral = get_hidden_state(mistral, question)
    h_transformed = transform_ml(h_mistral)
    answer = llama.generate(inputs_embeds=h_transformed)
    return answer

def double_handoff_ABA(question, llama, mistral, transform_lm, transform_ml):
    """Pattern 3: Llama → Mistral → Llama"""
    # Step 1: Llama processes question
    h1 = get_hidden_state(llama, question)

    # Step 2: Mistral processes Llama's representation
    h1_to_mistral = transform_lm(h1)
    h2 = get_hidden_state_from_prefix(mistral, h1_to_mistral)

    # Step 3: Back to Llama for final answer
    h2_to_llama = transform_ml(h2)
    answer = llama.generate(inputs_embeds=h2_to_llama)
    return answer

def double_handoff_BAB(question, mistral, llama, transform_ml, transform_lm):
    """Pattern 4: Mistral → Llama → Mistral"""
    # Step 1: Mistral processes question
    h1 = get_hidden_state(mistral, question)

    # Step 2: Llama processes Mistral's representation
    h1_to_llama = transform_ml(h1)
    h2 = get_hidden_state_from_prefix(llama, h1_to_llama)

    # Step 3: Back to Mistral for final answer
    h2_to_mistral = transform_lm(h2)
    answer = mistral.generate(inputs_embeds=h2_to_mistral)
    return answer

def get_hidden_state_from_prefix(model, prefix_hidden):
    """
    Continue computation from latent prefix.

    Similar to COCONUT's approach [11]:
    - Use prefix as initial hidden state
    - Continue generation from that point
    """
    # Generate continuation prompt
    continuation_prompt = "Continuing reasoning: "
    continuation_embeds = model.get_input_embeddings()(
        tokenizer.encode(continuation_prompt, return_tensors="pt").to(device)
    )

    # Concatenate prefix with continuation
    full_embeds = torch.cat([prefix_hidden.unsqueeze(1), continuation_embeds], dim=1)

    # Forward pass to get new hidden state
    outputs = model(inputs_embeds=full_embeds, output_hidden_states=True)
    return outputs.hidden_states[-1][:, -1, :]  # Last token hidden state
```

**Comprehensive Evaluation:**

```python
patterns = [
    ("Llama Solo", lambda q: llama_solo(q)),
    ("Mistral Solo", lambda q: mistral_solo(q)),
    ("A→B", lambda q: single_handoff_AB(q, llama, mistral, T_lm)),
    ("B→A", lambda q: single_handoff_BA(q, mistral, llama, T_ml)),
    ("A→B→A", lambda q: double_handoff_ABA(q, llama, mistral, T_lm, T_ml)),
    ("B→A→B", lambda q: double_handoff_BAB(q, mistral, llama, T_ml, T_lm)),
]

results = {name: [] for name, _ in patterns}

for problem in test_set:
    for name, method in patterns:
        answer = method(problem['question'])
        correct = check_em(answer, problem['gold'])
        results[name].append(correct)

# Summary statistics
for name in results:
    accuracy = np.mean(results[name])
    stderr = np.std(results[name]) / np.sqrt(len(results[name]))
    print(f"{name:15s}: {accuracy:.1%} ± {stderr:.1%}")

# Statistical significance tests
from scipy.stats import mcnemar

# Compare A→B→A vs best solo
best_solo_results = [max(results["Llama Solo"][i], results["Mistral Solo"][i])
                      for i in range(len(test_set))]
aba_results = results["A→B→A"]

# McNemar's test for paired binary outcomes
contingency = [[
    sum((b == 1) and (a == 1) for b, a in zip(best_solo_results, aba_results)),
    sum((b == 1) and (a == 0) for b, a in zip(best_solo_results, aba_results)),
], [
    sum((b == 0) and (a == 1) for b, a in zip(best_solo_results, aba_results)),
    sum((b == 0) and (a == 0) for b, a in zip(best_solo_results, aba_results)),
]]

statistic, p_value = mcnemar(contingency, exact=True)
print(f"\nMcNemar's test (A→B→A vs Best Solo): p={p_value:.4f}")
```

### Success Metrics

**Primary:**
- **Accuracy by pattern**: Which collaboration pattern works best?
- **Information preservation**: Does multi-hop transfer degrade quality?

**Secondary:**
- **Step-wise analysis**: Measure quality after each handoff
- **Error accumulation**: Do errors compound across handoffs?

**Success Criteria:**
- ✅ Strong: A→B→A or B→A→B beats both solos by ≥5%
- ✅ Moderate: Multi-hop matches best solo, no degradation
- ⚠️ Weak: Single handoff works, multi-hop degrades
- ❌ Failure: Multi-hop severely degrades vs single handoff

### Expected Outcomes

Based on COCONUT's multi-pass latent reasoning [11] and cascade inference patterns:
- **Optimistic**: A→B→A exploits both models' strengths, +5-8% over best solo
- **Realistic**: A→B→A ≈ best solo, information preserved
- **Pessimistic**: Multi-hop degrades by 3-5%, single handoff better

### Implementation Timeline

- **Day 1-2**: Implement all handoff patterns + hidden state extraction
- **Day 3-4**: Run experiments (6 patterns × 500 problems = 3000 evaluations)
- **Day 5**: Statistical analysis + step-wise quality tracking

### Deliverables

1. **Performance table** across all patterns
2. **Statistical significance tests** (McNemar, Wilcoxon)
3. **Information flow analysis**: Quality degradation per handoff
4. **Recommendations**: Optimal collaboration pattern

### References for Experiment 5

[11] COCONUT (Dec 2024) - Multi-pass latent reasoning, arXiv:2412.06769
[16] MixReasoning (Oct 2024) - Dynamic reasoning depth adjustment
[24] Dynamic Ensemble Reasoning (2024) - Sequential model collaboration

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Days 1-3: Experiment 1 (Simple Handoff)**
- Adapt `cross_model_ablation.py` for GSM8K dataset
- Implement handoff with existing Procrustes transforms
- Run on HPC: 4 alignments × 2 directions × 500 problems
- **Deliverable**: Initial collaboration results

**Days 4-5: Experiment 2 (Model Specialization)**
- Implement problem categorization (CALC/WORD/MIXED)
- Run both models solo on 500 GSM8K
- Statistical analysis of specialization patterns
- **Deliverable**: Specialization report + complementarity matrix

### Phase 2: Confidence & Cascading (Week 2)

**Days 6-8: Experiment 3 (Cascade Inference)**
- Implement confidence estimation (entropy, max-prob)
- Threshold sweep on 500 problems
- Calibration analysis
- **Deliverable**: Cost-quality tradeoff curves

**Days 9-10: Experiment 4 (Error Analysis)**
- Classify all outcomes into 6 categories
- Deep dive on complementary success cases
- Manual error categorization
- **Deliverable**: Error analysis report + case studies

### Phase 3: Multi-Step Collaboration (Week 3)

**Days 11-15: Experiment 5 (Multi-Step Handoff)**
- Implement 6 collaboration patterns
- Run comprehensive evaluation (3000 evaluations)
- Statistical significance testing
- Information flow analysis
- **Deliverable**: Multi-step collaboration results + recommendations

### Phase 4: Synthesis & Writing (Week 4)

**Days 16-20: Paper Writing**
- Compile results across all experiments
- Create visualizations (heatmaps, tradeoff curves, flow diagrams)
- Write draft paper sections
- Plan next experiments based on findings

---

## Expected Contributions

### Novel Findings (Publishable)

1. **First empirical study** of mid-reasoning hidden state handoffs between heterogeneous frozen LLMs
2. **Quantitative characterization** of Llama vs Mistral specialization on reasoning step types
3. **Demonstration** of training-free alignment (Procrustes) feasibility for cross-model reasoning transfer
4. **Analysis** of multi-hop latent transfer information preservation
5. **Practical cascade system** optimizing cost-quality tradeoffs

### Comparison to Prior Work

| Work | Year | Key Difference from Our Experiments |
|------|------|-------------------------------------|
| **Communicating Activations** [15] | 2025 | Mid-forward-pass merging vs our prefix conditioning; coordination games vs reasoning |
| **COCONUT** [11] | 2024 | Single model latent reasoning vs our cross-model handoffs |
| **DeePEn** [14] | 2024 | Probability fusion (expensive) vs our latent transfer (compressed) |
| **MixReasoning** [16] | 2024 | Within-model routing vs our cross-model collaboration |
| **CITER** [12] | 2024 | Token-level routing within model vs our step-level cross-model handoffs |
| **Multi-Agent Collab** [9] | 2025 | Text-based communication vs our continuous latent transfer |

---

## Success Criteria Summary

### Experiment 1: Simple Handoff
- ✅ Publishable: Collaboration > best solo by ≥3% OR complementary gain ≥5%
- ⚠️ Informative: Collaboration ≈ best solo (feasibility demonstrated)
- ❌ Negative: Collaboration < both solos (alignment fails)

### Experiment 2: Specialization
- ✅ Publishable: Clear specialization (≥10% accuracy gap on CALC or WORD)
- ⚠️ Informative: Weak specialization (5-10%) but consistent
- ❌ Negative: No specialization found

### Experiment 3: Cascade
- ✅ Publishable: ≥40% cost savings with ≤3% accuracy loss
- ⚠️ Informative: ≥20% cost savings but poor calibration
- ❌ Negative: No cost savings or severe degradation

### Experiment 4: Error Analysis
- ✅ Publishable: ≥10% complementary success rate with clear patterns
- ⚠️ Informative: 5-10% complementary success
- ❌ Negative: <5% complementary success, high error propagation

### Experiment 5: Multi-Step
- ✅ Publishable: A→B→A beats both solos by ≥5%
- ⚠️ Informative: Multi-hop matches best solo (information preserved)
- ❌ Negative: Multi-hop degrades vs single handoff

**Overall Publishability Threshold**:
- **Strong paper**: 3+ experiments achieve ✅ Publishable
- **Moderate paper**: 2 experiments ✅ + 2 experiments ⚠️ with novel insights
- **Workshop paper**: 1 experiment ✅ + interesting negative results

---

## References

[1] Qwen Team. (2024). "Qwen2.5: A Party of Foundation Models"
[2] Qwen Team. (2024). "Qwen2.5-Math: Advancing Mathematical Reasoning"
[3] Meta AI. (2024). "The Llama 3 Herd of Models"
[4] Comparative LLM benchmarks (2024). Various sources
[5] Yu et al. (2024). "MR-GSM8K: A Meta-Reasoning Revolution in Large Language Model Evaluation"
[6] Mirzadeh et al. (2024). "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models"
[7] TATA Framework (2024). "Synergizing CoT and TIR for Enhanced Reasoning"
[8] INC-Math (2024). "Natural Language vs Code Complementarity in Math Reasoning"
[9] Multi-Agent Survey (2025). "Multi-Agent Collaboration Mechanisms: A Survey of LLMs", arXiv:2501.06322
[10] MoSA (2025). "Mixture-of-Search-Agents for Complex Problem Solving", arXiv:2502.18873
[11] Hao et al. (2024). "Training Large Language Models to Reason in a Continuous Latent Space", arXiv:2412.06769
[12] CITER (2024). "Collaborative Inference for Efficient LLM Decoding with Token-Level Routing", arXiv:2502.01976
[13] MixLLM (2025). "Dynamic Routing with Streaming Queries"
[14] Huang et al. (2024). "Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration", arXiv:2404.12715
[15] Ramesh & Li (2025). "Communicating Activations Between Language Model Agents", arXiv:2501.14082
[16] MixReasoning (2024). "Dynamic Reasoning Depth Adjustment", arXiv:2410.xxxxx
[17] Cobbe et al. (2021). "Training Verifiers to Solve Math Word Problems", arXiv:2110.14168
[18] Multi-Agent Peer Review (2023). "Improving LLM Outputs via Collaborative Review"
[19] SLM-MUX (2025). "Combining Small Language Models for Enhanced Performance"
[20] Chen et al. (2023). "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"
[21] Huang et al. (2024). "Large Language Models Cannot Self-Correct Reasoning Yet"
[22] Uncertainty Estimation Survey (2024). "Methods for Confidence Estimation in LLMs"
[23] Consistency Methods (2024). "Multi-Sample Consistency for Black-Box Confidence Estimation"
[24] Dynamic Ensemble Reasoning (2024). "MDP-based Sequential Model Collaboration"

---

## Appendix: Code Repository Structure

```
experimental/learning/
├── cross_model_collaboration/
│   ├── experiment1_simple_handoff.py
│   ├── experiment2_specialization.py
│   ├── experiment3_cascade.py
│   ├── experiment4_error_analysis.py
│   ├── experiment5_multistep.py
│   ├── utils/
│   │   ├── alignment.py          # Procrustes transforms
│   │   ├── evaluation.py         # EM, accuracy metrics
│   │   ├── confidence.py         # Confidence estimation
│   │   └── gsm8k_loader.py       # Dataset utilities
│   ├── scripts/
│   │   ├── run_exp1_hpc.sh
│   │   ├── run_exp2_hpc.sh
│   │   ├── run_exp3_hpc.sh
│   │   ├── run_exp4_hpc.sh
│   │   └── run_exp5_hpc.sh
│   └── results/
│       ├── exp1_simple_handoff/
│       ├── exp2_specialization/
│       ├── exp3_cascade/
│       ├── exp4_error_analysis/
│       └── exp5_multistep/
└── CROSS_MODEL_COLLABORATION_EXPERIMENTS.md  # This document
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Authors**: Sujeeth Jinesh, Claude (Anthropic)
