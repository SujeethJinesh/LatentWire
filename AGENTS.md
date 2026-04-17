# Agent Guidance

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
