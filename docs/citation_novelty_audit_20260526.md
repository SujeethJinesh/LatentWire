# SA-LIT Citation and Novelty Audit

Date: 2026-05-26
Scope: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex` plus bibliography context. This audit did not edit the paper, bibliography, release, swarm, or experiment files.

## Executive Status

No CRITICAL hallucinated or source-missing prior-art citation was found in the requested focus set. The main risk is wording strength: several claims are directionally supported but should be softened because the cited paper's measurement surface, task setting, or quantization target differs from SA-LIT's.

Paper readiness impact: citation posture is close to workshop-ready after wording fixes, but not ICLR-ready as a positive-method paper. The live story remains a scoped mechanism paper: long-decode channel-set drift weakens static protected-channel assumptions; budget-tuned EMA is partial and model-dependent; rotations are a stronger comparator mechanism. Blocking gap for top-tier submission remains stronger positive-method evidence, not citation provenance.

## CRITICAL Findings

None found in the scoped prior-art audit.

I did not find a fabricated DecDEC, Quamba2, ParoQuant, SLQ, QEP, LAQuant, ChanMix, MixKVQ, QMamba, OuroMamba, MambaQuant, AWQ, SmoothQuant, QuaRot, KVQuant, KIVI, or PM-KVQ citation. All resolved to primary sources or official venue/model pages.

## SUBSTANTIAL Findings

### SUBSTANTIAL: "Contradicting Quamba2" is too strong outside the measured surface

- Paper wording: "contradicting the channel-persistence assumption behind Quamba2-style static ordering"; "show that Quamba2's channel-persistence assumption fails at the measured block-output surface".
- Source URLs:
  - https://arxiv.org/abs/2503.22879
  - https://openreview.net/pdf/61e45e9890791f2c4b78cea7d0a7a7ce4984a4c5.pdf
- Source support: Partially supported. Quamba2 explicitly bases its offline sorting/clustering on "channel order preserving and activation persistence of SSMs" and targets W8A8, W4A8, and W4A16 for Mamba1/Mamba2. However, Quamba2's cited assumption is about the inputs of a linear recurrence and SSM quantization internals, while SA-LIT measures accessible block-output activation channels, not every internal SSM state/input tensor.
- Risk: A reviewer can read "contradicting Quamba2" as claiming direct falsification of Quamba2's internal recurrence-input assumption.
- Recommended wording: "At the measured block-output surface, our long-decode traces challenge the protected-set persistence one would need when applying Quamba2-style static ordering to these hybrid-model outputs; we do not claim to falsify every internal SSM tensor analyzed by Quamba2."

### SUBSTANTIAL: DecDEC four-axis novelty should separate extension from overlap

- Paper wording: "Our novelty is horizon, model class, architecture, and reasoning benchmark surface."
- Source URLs:
  - https://www.usenix.org/conference/osdi25/presentation/park-yeonhong
  - https://www.usenix.org/system/files/osdi25-park-yeonhong.pdf
- Source support: Partially supported. DecDEC dynamically identifies salient activation channels at each decode step and explicitly motivates this by changing activation distributions during decode. SA-LIT extends the horizon and adds hybrid Mamba/parallel-hybrid coverage, but the paper also includes a pure-Transformer control, so "architecture" and "model class" are not cleanly independent novelty axes for every reported model.
- Risk: A reviewer may see this as overclaiming novelty relative to DecDEC rather than a measured extension.
- Recommended wording: "Relative to DecDEC's short Transformer decode setting, our extension is longer reasoning horizons, strict set-membership drift accounting, hybrid Mamba/parallel-hybrid coverage, and reasoning-trace intervention controls; the pure-Transformer result is a falsification/control surface rather than a separate architectural novelty claim."

### SUBSTANTIAL: SLQ compound-error sentence is not directly supported as written

- Paper wording: "SLQ finds that near-lossless quantization errors need not compound across token positions."
- Source URL: https://arxiv.org/abs/2605.02404
- Source support: Weak/partial. The SLQ abstract supports task-lossless and distribution-lossless compression, next-token distribution fidelity via Expected Acceptance Rate, and task/distribution preservation at specific bit budgets. It does not, on the primary arXiv page, directly make the token-position compound-error claim in the form used here.
- Risk: This is the most likely "citation says more than the source" issue in the current related-work text.
- Recommended wording: "SLQ shows that quantized LLMs can preserve task-level and next-token distribution fidelity under sufficiently careful bit allocation; this motivates testing whether our more aggressive W4A16 endpoint exhibits decode-position error growth rather than assuming it."

### SUBSTANTIAL: ParoQuant should be scoped as weight-only/rotation PTQ unless citing SA-LIT's own baseline

- Paper wording: "Recent W4A16 systems ... including ParoQuant's rotation-based reasoning results"; "ParoQuant reports strong W4A16 recovery for reasoning models through pairwise rotations."
- Source URLs:
  - https://arxiv.org/abs/2511.10645
  - https://openreview.net/forum?id=1USeVjsKau
- Source support: Mostly supported, but terminology should be exact. ParoQuant is described as a PTQ method using pairwise/Givens rotations and channel-wise scaling; the OpenReview page says "Under weight-only quantization" it improves over AWQ on reasoning tasks and matches state-of-the-art weight-activation quantization methods. The source does not frame its headline as SA-LIT-style "recovery" of a BF16-vs-static protected-channel gap.
- Risk: "W4A16 recovery" can blur source-reported ParoQuant results with SA-LIT's own Granite baseline result.
- Recommended wording: "ParoQuant reports strong reasoning-task accuracy under weight-only low-bit PTQ using pairwise rotations and channel-wise scaling; our Granite ParoQuant-style rotation baseline then tests whether a rotation mechanism, rather than a channel-set protection mechanism, can recover the SA-LIT endpoint."

## MINOR Findings

### MINOR: Mamba dynamic-outlier prior-art boundary is handled well

- Paper wording: "Vision-Mamba quantization already reports dynamic activation outliers"; "do not claim this is the first dynamic-outlier finding in Mamba" in reviewer context.
- Source URLs:
  - QMamba: https://arxiv.org/abs/2501.13624
  - OuroMamba: https://arxiv.org/abs/2503.10959
  - Mamba-PTQ: https://arxiv.org/abs/2407.12397
  - MambaQuant: https://arxiv.org/abs/2501.13484
- Source support: Supported. QMamba reports highly dynamic hidden-state sequence variation in vision SSMs. OuroMamba reports dynamic outlier variations across time steps and dynamic outlier detection during inference. Mamba-PTQ identifies activation outliers in recurrent LLMs. MambaQuant uses variance-aligned rotations for Mamba-family quantization.
- Recommended wording: Keep current stance. Avoid any "first dynamic outlier" wording.

### MINOR: Static-protection systems are heterogeneous; current caveat is good but the grouping can be sharper

- Paper wording: "SmoothQuant, AWQ, QuaRot, KVQuant, and BlockDialect protect or transform activation structure using calibration-time statistics, rotations, or adaptive formats."
- Source URLs:
  - SmoothQuant: https://arxiv.org/abs/2211.10438
  - AWQ: https://arxiv.org/abs/2306.00978
  - QuaRot: https://arxiv.org/abs/2404.00456
  - KVQuant: https://arxiv.org/abs/2401.18079
  - KIVI: https://arxiv.org/abs/2402.02750
- Source support: Supported with nuance. SmoothQuant migrates activation quantization difficulty to weights. AWQ protects salient weight channels using offline activation statistics. QuaRot removes outliers through rotations. KVQuant targets KV-cache activations with key/value-specific methods, including per-vector dense-and-sparse handling.
- Recommended wording: "SmoothQuant, AWQ, QuaRot, KVQuant, and related systems use offline activation statistics, equivalent scaling, rotations, dense/sparse outlier handling, or adaptive formats to make low-bit inference viable. They are motivation, not contradicted baselines."

### MINOR: KV-cache comparator sentence is supported

- Paper wording: "KV-cache methods such as KIVI, PM-KVQ, AttentionPredictor, ChanMix, and MixKVQ adapt or compress cached key/value tensors; our interventions target linear-layer activation channels."
- Source URLs:
  - KIVI: https://arxiv.org/abs/2402.02750
  - PM-KVQ: https://arxiv.org/abs/2505.18610
  - AttentionPredictor: https://arxiv.org/abs/2502.04077
  - ChanMix: https://openreview.net/forum?id=yjr2jX41qO
  - MixKVQ: https://arxiv.org/abs/2512.19206
- Source support: Supported. These sources target KV-cache compression/quantization or attention/KV criticality, not SA-LIT's linear-layer activation-channel protection.
- Recommended wording: Keep current distinction.

### MINOR: QEP/LAQuant/Activation Sensitivity are relevant, but split their claims

- Paper wording: "QEP, LAQuant, and Activation Sensitivity emphasize layer-wise propagation, lookahead losses, calibration mismatch, and task-conditional sensitivity as PTQ risks."
- Source URLs:
  - QEP: https://arxiv.org/abs/2504.09629
  - LAQuant: https://arxiv.org/abs/2605.08755
  - Activation Sensitivity: https://arxiv.org/abs/2601.11663
- Source support: Mostly supported. QEP supports layer-wise error accumulation and compensation. LAQuant supports layer-wise lookahead loss plus calibration/deployment-distribution alignment concerns. Activation Sensitivity supports channel-wise perturbation impact on loss and downstream error propagation, but "task-conditional sensitivity" is an inference rather than the source's headline wording.
- Recommended wording: "QEP emphasizes cross-layer quantization-error growth, LAQuant emphasizes layer-wise lookahead losses and calibration/deployment alignment for reasoning models, and Activation Sensitivity formalizes channel-wise perturbation impact on loss and downstream propagation."

### MINOR: ParoQuant inline bibliography appears to omit one arXiv author

- Paper wording: inline bibliography lists "Yesheng Liang, Haisheng Chen, Song Han, and Zhijian Liu."
- Source URLs:
  - https://arxiv.org/abs/2511.10645
  - https://openreview.net/forum?id=1USeVjsKau
- Source support: The arXiv page lists Yesheng Liang, Haisheng Chen, Zihan Zhang, Song Han, and Zhijian Liu. OpenReview lists four authors and omits Zihan Zhang. Because the TeX cites the arXiv URL, the safer bibliography should match arXiv.
- Recommended wording: In a future bibliography-only pass, use the arXiv author list or cite OpenReview consistently.

### MINOR: TechCrunch bibliography byline mismatch is outside the requested prior-art focus but should be cleaned later

- Paper wording: inline bibliography uses `Bort(2025)` / Julie Bort.
- Bibliography context: `bibliography.bib` already notes that the retained key maps to a TechCrunch page by Kyle Wiggers.
- Source URL: https://techcrunch.com/2025/04/10/the-rise-of-ai-reasoning-models-is-making-benchmarking-more-expensive/
- Source support: This is not a scientific novelty risk, but it is a citation hygiene issue.
- Recommended wording: In a future bibliography-only pass, update the inline bibliography author to Kyle Wiggers or remove commercial forecasts if venue page pressure is tight.

## Source Coverage Checklist

- DecDEC: verified against USENIX page and OSDI PDF. Supported for dynamic per-step salient-channel identification; static-short-horizon recall wording should remain scoped.
- Quamba2: verified against arXiv/OpenReview PDF. Supported for channel-order preservation/activation persistence as a Quamba2 assumption; direct contradiction should remain limited to SA-LIT's measured block-output surface.
- ParoQuant: verified against arXiv and OpenReview. Supported for pairwise rotations, channel-wise scaling, reasoning-task gains, and weight-only low-bit PTQ; avoid mixing source-reported results with SA-LIT recovery metric.
- SLQ: verified against arXiv. Supported for task/distribution-lossless quantization; not directly enough for the current token-position compounding sentence.
- QEP/LAQuant/Activation Sensitivity: verified against arXiv. Relevant to propagation/lookahead/sensitivity, with wording refinements above.
- QMamba/OuroMamba/Mamba-PTQ/MambaQuant: verified against arXiv. Current non-first dynamic-outlier boundary is appropriate.
- AWQ/SmoothQuant/QuaRot/KVQuant/KIVI/PM-KVQ/ChanMix/MixKVQ: verified against arXiv/OpenReview. Current high-level comparator distinction is supported if kept heterogeneous and non-defeating.

## Highest-Priority Wording Patch Later

If the paper is edited after this audit, the first patch should address only three sentences:

1. Replace "contradicting Quamba2" with "challenging Quamba2-style protected-set persistence at the measured block-output surface."
2. Replace the DecDEC novelty sentence with the scoped extension wording above.
3. Replace the SLQ sentence with a task/distribution-fidelity motivation rather than a direct token-position compounding claim.
