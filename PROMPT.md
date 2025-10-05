Please review the RESEARCH_PROPOSAL.md, LOG.md and paper.tex. This is for the
project we're working on that we want to submit to MLSys or Neurips. Then go
through the papers folder to learn about other references that may be relevant.
Then go through all the code in latentwire/ and scripts folders to understand the
exact code. Once that's all done, I just completed a smoke test and the logs are
in runs/smoke. Analyze the logs in runs/smoke that was run by the
scripts/run_llama_single.sh. Then determine what happened, what's good, what's
bad, and what are our next steps. Always commit and push your code when done.
When appropriate, please update LOG.md with why a change was made, evidence for it,
what was expected, and the result with data after we analyze the next set of runs.
For larger architectural changes, we should always update the RESEARCH_PROPOSAL.md.
I also highly prefer you not make new scripts unless necessary and just modify existing ones. Also
prefer keeping logic into one script over many. It may be worth writing python scripts instead of bash scripts to make things much easier or converting them

---

please read through the papers folder and read through
possible_improvements/possible_improvements.md. These are potential updates we may want to read
through if our current approach fails. The conversation in that md file was with ChatGPT and is used
to find possible ways of making our interlingua work. Based off our current results,
and the conversation, please identify what may be a reasonable architectural change we can do
and if you agree with ChatGPT. If you think it's reasonable to do so, then come up with your plan
to update our code based on the best next step.
