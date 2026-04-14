# Agent Guidance

- Use subagents liberally when the environment supports them and the task benefits from parallel exploration, implementation, or verification.
- Cite sources frequently, especially when making claims about prior work, benchmark design, baselines, ablations, or theoretical motivation.
- Optimize for novel contribution rather than incremental replication. Prefer experiments and analyses that sharpen what is new, defensible, and publishable.
- Hold the project to ICLR / NeurIPS quality. That means strong baselines, rigorous ablations, clear threat-modeling of limitations, and evidence that the claimed contribution survives reviewer scrutiny.
- When choosing between shortcuts and rigor, bias toward the version that would stand up in a top-tier conference submission.
- Use the repo-local virtual environment for Python work. Install Python dependencies only into `./.venv`, source that environment before Python commands when practical, do not install packages into the global interpreter, and keep all Python downloads and caches scoped to `./.venv`.
- Be brief in explanations back to the user. Default to concise, high-signal summaries unless the user asks for more depth.

# Debug folder

Create and use a .debug folder that you will not check in but should be used for your own experimentation. This is a scratch space for you to test our reproductions of bugs or issues so that way you can better fix problems in our production codebase.
