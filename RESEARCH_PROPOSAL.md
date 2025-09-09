Below is a ruthlessly scoped, 8‑week research proposal for an MVP that you can ship. It focuses on one core result: a compact, shared soft‑token interlingua that (i) matches or nearly matches text prompting at 4×+ compression, and (ii) delivers measurable 2‑LLM gains via a simple, deterministic joint‑rescoring combiner. Everything else is optional or a fallback.

0. MVP definition (what we will ship)

Problem: Enable two heterogeneous LLMs (Llama & Qwen) to consume the same compact latent prefix with ≥4× prompt compression versus text, with task quality near text and two‑model gains over either model alone.

MVP success criteria (hard gates):

G1 (Quality @ Compression): On HotpotQA‑val, Latent F1 ≥ Text F1 − 2.0 pts at ≥4× compression (vs each model’s avg prompt tokens).

G2 (Synergy): Joint pick (equal‑weight sum of log‑probs with temperature calibration) ≥ +1.5 pts F1 over the better single‑model latent.

G3 (Conditioning): NLL/token (gold) under latent ≤ 1.05× text.

G4 (Efficiency): Prefill token count reduced ≥4× and wall‑clock (prefill+decode) non‑worse than text baseline on the same hardware.

Primary deliverables:

Code (already scaffolded) with train.py, eval.py, infer.py and the interlingua modules.

Tables/plots: F1/NLL/token vs M, compression & latency, joint pick vs best single model, token‑budget baseline vs latent.

Paper draft: abstract + intro + methodology + results + systems analysis (week 8).

Non‑goals (unless ahead of schedule):

Discrete codebook (VQ) output channel.

Multi‑round message passing.

Extensive cross‑task generalization.

1. Scope, models, and resources

Models (default): TinyLlama/TinyLlama-1.1B-Chat-v1.0 and Qwen/Qwen2-0.5B-Instruct (fit on 24–40 GB GPU with batch=1).
Stretch: Llama‑3.1‑8B‑Instruct + Qwen2‑7B‑Instruct (with --load_4bit).

Dataset: HotpotQA (fullwiki or distractor fallback). Use train: 10k, val: 1k, test: 1k (subsampled for speed).

Hardware: 1× A100‑40GB (or similar). If only 24GB, keep to 1.1B/0.5B and batch=1.

Runtime expectations: Training (encoder+adapters) is light parameter‑wise but backprops through LLMs. With small models: 2–5 hrs per sweep (M∈{6,8,12,16}) at batch=1.

2. Work breakdown (must‑have vs nice‑to‑have)

Must‑have (MVP core):

M1. Capacity sweep: M ∈ {6,8,12,16}; pick the smallest M that satisfies G1–G3.

M2. Token‑budget baseline (already added): textual prompt truncated to M tokens.

M3. Calibration + joint pick: temperature scaling per model on val; equal‑weight sum on test; report F1 gains.

M4. Diversity/coverage regularizer for cross‑attn queries (lightweight).

M5. Asymmetric latent (shared + per‑model) if shared‑only misses G1 by >2 pts at M≤16.

Should‑have (only if time permits):

S1. Learned aggregator (logistic regression over simple features) vs equal‑weight joint pick.

S2. Latency microbench: separate prefill vs decode wall‑clock; approximate FLOPs.

Nice‑to‑have / fallback:

N1. LoRA‑early (rank 4–8, first 2–3 layers Q/V) if inputs remain unstable.

N2. Hybrid front‑end (bytes + tiny subword lexicon) if byte encoder underperforms.

3. Week‑by‑week plan (8 weeks; week 8 is paper)
   Week 1 — Bring‑up, baselines, first sweep (MVP skeleton)

Goals: End‑to‑end runs; text vs latent vs token‑budget for M∈{8,12}.
Tasks:

✅ Environment pin + HF auth; confirm train.py/eval.py run on small subset (128 samples).

✅ Train with M=8,12 on train=5k, evaluate on val=1k.

✅ Produce baseline table (Text / Token‑budget / Latent) for EM/F1, NLL/token, compression, latency.
Acceptance: Latent beats token‑budget at same M by ≥+3 F1 (or clear path identified). If not, raise M and continue.

Week 2 — Capacity & stability sweeps (hit G1–G3 at small M if possible)

Goals: Lock in an M ≤ 16 satisfying G1–G3.
Tasks:

Add coverage regularizer to pooler; sweep M=6,8,12,16 on train=10k.

Add input regularization (tanh clipping + mean/var match of prefix vs token embeddings).

Re‑run; choose best M.
Acceptance (Gate G1–G3): If still short, enable Asymmetric latent (shared=4, model‑specific=4) and rerun M=8,12.

Week 3 — Two‑model synergy (G2)

Goals: Show Joint pick ≥ +1.5 F1 over best single‑latent.
Tasks:

Calibration: temperature scaling per model (on val).

Implement equal‑weight joint pick; report on test.

If needed, add S1 Learned aggregator (logistic regression features: logP_L, logP_Q, length, repetition score, agreement).
Acceptance (Gate G2): If synergy < +1.5, enable Asymmetric latent (if not already) and/or increase M one notch (bounded by 16). If still low, consider one‑round message head (optional) only if minimal code change.

Week 4 — Efficiency & system results (G4)

Goals: Show compression ≥4× and non‑worse wall‑clock than text.
Tasks:

Time prefill (prefix forward) and decode separately (simple timing hooks).

Report compression, payload bytes, prefill tokens, latency.

If wall‑clock worse, profile and reduce ByteEncoder cost (strided attention or conv downsample) without hurting F1.
Acceptance (Gate G4): Non‑worse wall‑clock at chosen M. If not, accept minor quality drop to meet wall‑clock, but keep G1 within −2.5 pts.

Week 5 — Hardening & minimal ablations

Goals: Produce minimal but convincing ablations for paper.
Tasks:

F1 vs M (6,8,12,16) curve (final settings).

Shared vs Asymmetric at fixed total M.

With vs without coverage loss.

Token‑budget vs Latent at same M.

Save seeds, configs, logs; generate CSVs.

Week 6 — Fallbacks only if needed

Only if gates still unmet:

Enable LoRA‑early (rank=4–8 on first 2–3 layers Q/V).

Small LR sweep + alternating schedule (encoder+L, encoder+Q).

Re‑run Week‑2/3 experiments with small LoRA.

Week 7 — Freeze, replicate, and test

Goals: Freeze best config; replicate 3 runs for CIs.
Tasks:

Run 3 seeds; aggregate mean ± 95% CI (bootstrap) for headline metrics.

Finalize plots & tables.

Create a make_figs.py to render plots from CSV.

Week 8 — Paper only

Goals: Write; no new experiments unless fatal bug.
Tasks:

Results & discussion; systems analysis; limitations; related work.

Append reproducibility checklist; release code + configs.

4. Concrete task list (ticketized)

T1. Coverage regularizer (R6)

Modify LatentPooler to return attention maps and add diversity_loss (weight 0.01).

Acceptance: +≥0.5 F1 at M=8 vs no regularizer, or same F1 with lower M.

T2. Input regularization (R3)

Adapter: LayerNorm → Linear → tanh(x/3)\*3; prefix mean/var match loss (weight 1e‑3).

Acceptance: latent NLL/token improves ≥5% where it lagged.

T3. Asymmetric latent (R2/R7)

Produce Z_shared + Z_L + Z_Q under fixed total M.

Acceptance: +≥1.5 F1 over shared‑only at same M if shared failed G1.

T4. Calibration + joint pick (R9)

Temperature scaling on val, equal‑weight sum on test.

Optional: logistic aggregator (S1) if equal‑sum < +1.5 F1.

Acceptance: G2 satisfied.

T5. Efficiency hooks (R10)

Time prefill vs decode; log prefill tokens & payload bytes.

Acceptance: G4 satisfied.

(Fallback) T6. LoRA‑early (R3)

Add PEFT LoRA on first 2–3 layers (Q/V).

Acceptance: meets G1–G3 when others fail.

5. Experiment matrix (minimal)
   Exp ID Purpose Config Report
   E1 Capacity / choose M M∈{6,8,12,16}; coverage=on; reg=on EM/F1, NLL/token, compression, latency
   E2 Shared vs Asym M fixed (best); split shared/LLM‑specific EM/F1, NLL/token
   E3 Synergy Best M; joint pick (equal‑sum) F1 gain vs best single latent
   E4 Token‑budget fairness Text truncated to M vs Latent EM/F1
   E5 Efficiency Prefill tokens, payload bytes, wall‑clock Speed & memory narrative

(Only if needed) E6: LoRA‑early vs none at best M.

6. Risks & contingency (aligned to the 10 failure modes)

If G1 fails at M≤16: switch to Asymmetric; if still fails, raise M to 24 with explicit note that we’re trading some compression for quality.

If G2 fails: add calibration (temp scaling) and a learned aggregator; if still low, add one‑round message head (2 vectors) only if trivial to wire.

If NLL/token high or outputs degenerate: ensure input regularization on; then LoRA‑early.

If wall‑clock worse: add ByteEncoder stride or smaller d_z; accept −0.5 F1 if needed to win efficiency.

7. Data, logging, and reproducibility

Fix seeds; pin versions in requirements.txt.

Standard splits: train=10k, val=1k, test=1k (stratified random).

Save all run configs + metrics to CSV/JSON, plus model ckpts for the tiny modules.

Provide scripts/{train,eval}\_runner.sh to replicate all tables.

8. Paper plan (Week 8)

Figures:

F1 vs M; NLL/token vs M; latency vs M; shared vs asymmetric; token‑budget vs latent; joint pick gain; compression & payload.

Tables:

Headline: Text vs Latent at chosen M (per model) + Joint pick; compression; latency.

Ablations: coverage reg on/off; asymmetric on/off.

Sections ready: Abstract, Intro, Methodology (already drafted); Results; Systems Analysis; Limitations.

9. Day‑to‑day cadence (practical)

Mon/Tue: implement + kick off sweeps overnight.

Wed: analyze, decide next M/toggle.

Thu: run confirmatory evals.

Fri: write/update a results log and checkpoint figures.

Always keep one best config runnable end‑to‑end.

10. Immediate next actions (this week)

Run Week‑1 plan today: M={8,12} with current code; create baseline table.

Implement T1 (coverage loss) and T2 (input regularization); re‑run M={6,8,12,16}.

Book a 4–8 hr block on a single GPU nightly for sweeps; keep batch=1; cap max_answer_tokens=32.

If you want, I can produce small code patches (a few functions) for coverage loss, asymmetric latent, and calibration + joint pick as drop‑ins to your repo so you can start Week‑1/2 immediately.
