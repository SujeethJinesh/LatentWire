# HellaSwag Hybrid Anti-Harm Veto References

Date: 2026-05-03

## Why This Memo Exists

The anti-harm veto gate tested whether a packet-preserving selective fallback
could improve the fixed HellaSwag hybrid packet. It failed. This memo records
the novelty boundary: the attempted veto is a selective-classification /
learning-to-defer router under a strict source-private packet contract, not a
new standalone abstention method.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: multiple-choice commonsense benchmark used by the strict same-family
     and cached cross-family gates.
   - Boundary: a candidate-id packet result remains multiple-choice scoped
     unless broader benchmarks are added.

2. Selective Classification for Deep Neural Networks
   - Link: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
   - Role: establishes risk-coverage style selective prediction for deep
     models.
   - Boundary: the LatentWire veto is not novel as selective prediction; any
     novelty would need to come from the fixed-byte source-private packet
     setting plus strict destructive controls.

3. SelectiveNet: A Deep Neural Network with an Integrated Reject Option
   - Link: https://arxiv.org/abs/1901.09192
   - Role: reject-option neural selective classifier baseline.
   - Boundary: our killed gate used a one-rule source-side selector, not a
     learned integrated reject model.

4. Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer
   - Link: https://papers.nips.cc/paper/7853-predict-responsibly-improving-fairness-and-accuracy-by-learning-to-defer
   - Role: formalizes choosing between model and expert decisions.
   - Boundary: packet-vs-hybrid fallback is a special defer/router case, so it
     cannot be claimed as new without the packet/privacy/rate contract.

5. Consistent Estimators for Learning to Defer to an Expert
   - Link: https://proceedings.mlr.press/v119/mozannar20b.html
   - Role: modern learning-to-defer formulation and consistency baseline.
   - Boundary: future target-aware acceptors should cite this line and report
     coverage/risk, not just accuracy.

6. On Calibration of Modern Neural Networks
   - Link: https://arxiv.org/abs/1706.04599
   - Role: calibration warning for confidence/margin-based vetoes.
   - Boundary: the no-op margin-only veto result is consistent with the need
     for calibrated uncertainty before using margins as accept/reject signals.

7. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: option-position and MCQ selector-bias threat.
   - Boundary: the failed rule `selected_id <= 1` is exactly the kind of
     option-position behavior reviewers will challenge; option-order controls
     remain required before broad reasoning claims.

8. PriDe: A Pseudo-label and Debiasing Framework for LLM Multiple-Choice
   Evaluation
   - Link: https://arxiv.org/abs/2308.11483
   - Role: MCQ debiasing and option-bias comparison point.
   - Boundary: future HellaSwag packet rows should either add option-order
     robustness or keep the claim narrowly as candidate-policy transfer.

9. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://openreview.net/forum?id=LeatkxrBCi
   - Role: high-rate source-state communication baseline.
   - Boundary: C2C fuses source KV/cache information. The veto remains a
     one-candidate packet and does not close the systems or expressivity gap.

10. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
    - Link: https://openreview.net/forum?id=F7rUng23nw
    - Role: selective KV-sharing communication comparator.
    - Boundary: KVComm transfers selected KV state. The killed veto sends no
      KV state and therefore needs native systems rows for fair systems claims.

11. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead
    - Link: https://arxiv.org/abs/2406.03482
    - Role: low-bit KV/vector compression floor.
    - Boundary: QJL pressures future vector-state LatentWire variants; it does
      not explain or replace a one-candidate discrete packet.

12. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: online vector quantization baseline for systems comparison.
    - Boundary: TurboQuant is relevant once LatentWire transmits vector/KV
      state or claims compression-native systems wins.

13. vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
    - Link: https://arxiv.org/abs/2309.06180
    - Role: serving substrate and TTFT/TPOT/goodput baseline.
    - Boundary: Mac byte accounting cannot replace native vLLM/SGLang style
      throughput, HBM, and scheduling measurements.

14. SGLang: Efficient Execution of Structured Language Model Programs
    - Link: https://arxiv.org/abs/2312.07104
    - Role: serving/runtime comparison point for structured LLM execution.
    - Boundary: the anti-harm veto does not justify serving-speed claims
      without native rows.

## Decision Boundary

This gate should be cited as:

```text
negative selective-router evidence under a fixed-byte source-private packet contract
```

It should not be cited as:

```text
new selective classification
new learning-to-defer
learned receiver fusion
common latent basis
native systems speedup
```

The next method branch should use the oracle gap to motivate target-loss
query/soft-prefix, conditional hidden-innovation, or sparse/common-basis
receiver methods rather than another shallow source-side veto.
