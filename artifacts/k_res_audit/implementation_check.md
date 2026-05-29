# K-RES Implementation Check

Status: `VALID_IMPLEMENTATION_KEEP_K_RES_PROXY_KILLED`.

1. Correct tight ParoQuant baseline: yes. `reference_run_dir` is `om_driftrot_granite_clip_tight_confirmation_20260528T1735Z` and the trace recovery reference is 5.940.
2. Same traces/windows: yes. Prompt index 4, scoring window 9489--10000, 512 scored tokens.
3. Rotated basis: yes for the implemented design. Delta columns are computed after applying the same ParoQuant-style rotation/quantization/dequantization/fold-back path used by the tight clip packet, then correction applies `W_fp - W_pq` input columns.
4. Shape/dtype/index ordering: pass at packet level. `installed_residual_corrections.json` records expert-bank mode, 8 modules, 32 columns/module, fp16 delta sidecars, and matching delta shapes.
5. Budget: exactly top-8 modules x 32 columns = 256 corrected input-column selections across expert-bank modules; total sidecar 54.0 MiB fp16.
6. Worsening scope: only one diagnostic tail trace was run, so concentration across traces is unknown. Within that trace the aggregate output loss worsened strongly.
7. KL/logit error: not available. The packet scored NLL/perplexity only and did not persist token-level logits or KL.
8. Residual energy vs loss benefit: yes, negative evidence. Residual-energy-selected columns had nontrivial residual score but produced much worse recovery than tight ParoQuant alone.

The first invalid run had a closure-capture bug and is excluded. The fixed run binds each module's original forward and sidecar tensors through a factory closure.
