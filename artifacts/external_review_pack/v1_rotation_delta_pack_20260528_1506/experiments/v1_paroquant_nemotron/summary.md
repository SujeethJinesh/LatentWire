# V1 ParoQuant-on-Nemotron

Hypothesis: ParoQuant may or may not transfer to Nemotron, where M11b top-10 was
the previous strongest positive method.

Method: ParoQuant-style scaled pairwise rotation plus W4A16 scoring, using the
same reused BF16/static trace baseline as Nemotron M11b.

Result: `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES`, median
1.047, CI95
[1.007, 1.292],
margin over M11b top-10 +0.232.

Inspect `per_trace.csv`, `check_script_output.txt`, and `source_score_cache/`.
