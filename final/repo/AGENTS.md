# Agent Guidance

## Standing ICLR Positive-Method Brief

We are working toward an ICLR-ready positive-method paper on cross-model
communication / latent transfer.

Your job is to keep pushing the project forward until we have a
benchmark-backed, reproducible, interpretable positive method. Do not switch
into paper-writing mode until the evidence is strong enough.

### Operating rules

- Always begin by stating:
  1. current paper readiness status
  2. current story of the paper
  3. what exact gap still blocks submission
- Treat this as a gated research loop, not open-ended exploration.
- Use reviewer feedback, collected telemetry, and the experiment ledger as
  first-class inputs every turn.
- Use subagents and web research aggressively, but only for targeted questions
  that materially help the next decision.
- Keep results interpretable and reusable for the eventual paper.
- Note clearly what is saturated, what is still alive, and what has become the
  highest-priority next branch.
- Avoid going in circles. Explicitly log what hypotheses are ruled out,
  weakened, revived, or promoted.

### Hard priorities

1. Prove or kill the current live method branch on the strongest available
   decision surface.
2. Improve evaluation quality before claiming progress:
   - larger frozen slices
   - seed repeats
   - paired uncertainty
   - oracle/headroom diagnostics
   - strict same-family vs cross-family separation
3. Only after that, widen to competitor and long-context benchmarks.
4. Only pursue new method branches if they are among the top 1-2 highest
   expected-value next moves.

### Current research goal

- Find a positive method that survives beyond the tiny smoke slice and is
  strong enough for an ICLR paper.
- We are not allowed to settle for a non-positive-method paper.

### Execution requirements each turn

- Read the latest telemetry and reviewer feedback before deciding what to do.
- Pick the single most important gate to clear this turn.
- Then do the work end-to-end:
  - targeted research
  - code changes
  - experiments
  - tests
  - ledger / memo updates
  - push if appropriate
- Use subagents for:
  - recent literature
  - competitor baselines
  - benchmark design
  - lateral inspirations from quantization, multimodal, routing, symmetry,
    transport, diffusion, etc.
- But keep the main line of work tightly focused.

### Decision policy

- Do not widen benchmark scope if the current live branch has not yet cleared:
  - larger frozen slice
  - seed stability
  - one strict cross-family falsification pair
- Do not spend much time on branches already shown to be saturated unless there
  is a crisp new reason.
- Prefer branches that can either:
  - beat the current live row
  - preserve the current live row while clearly improving bytes / latency /
    robustness
  - or generalize it cross-family

### Output requirements

- Start with current paper status and estimated distance to ICLR readiness.
- Then state exactly what you are doing this turn and why.
- End with:
  - what changed
  - what the new evidence says
  - what the next exact gate is
  - commit hash if pushed

- Use subagents liberally when the environment supports them and the task benefits from parallel exploration, implementation, or verification.
- Cite sources frequently, especially when making claims about prior work, benchmark design, baselines, ablations, or theoretical motivation.
- Optimize for novel contribution rather than incremental replication. Prefer experiments and analyses that sharpen what is new, defensible, and publishable.
- Hold the project to ICLR / NeurIPS quality. That means strong baselines, rigorous ablations, clear threat-modeling of limitations, and evidence that the claimed contribution survives reviewer scrutiny.
- When choosing between shortcuts and rigor, bias toward the version that would stand up in a top-tier conference submission.
- Use a repo-local virtual environment for Python work. Install Python dependencies only into that repo-local venv, source it before Python commands when practical, do not install packages into the global interpreter, and keep all Python downloads and caches scoped to that venv. On this machine, prefer a stable repo-local venv such as `./venv_arm64` if `./.venv` drifts or becomes inconsistent across fresh processes.
- Be brief in explanations back to the user. Default to concise, high-signal summaries unless the user asks for more depth.
- When running long experiment loops, work in iterative cycles:
  1. run the cheapest discriminative ablations first,
  2. analyze what changed,
  3. research the next best idea with primary sources,
  4. implement the highest-yield bounded change,
  5. rerun focused controls before widening.
- In overnight or deadline-driven loops, prefer resumable commands, append-only logs, and tracked readout notes that summarize what changed and what the new evidence means.
- When adding new literature, update the `references/` directory and its manifests together so later paper writing can cite the sources cleanly.
- Use web research for any claim about the latest models, papers, or recent methods; prefer official docs, arXiv, OpenReview, ACL Anthology, or project pages over tertiary summaries.
- Bias toward experiments that separate real source communication from zero-byte target-cache effects. If a run cannot resolve that distinction, it is lower priority than one that can.

# Debug folder

Create and use a .debug folder that you will not check in but should be used for your own experimentation. This is a scratch space for you to test our reproductions of bugs or issues so that way you can better fix problems in our production codebase.
