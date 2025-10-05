
# Patch Notes (CCoT)
- Add ThoughtHead to emit k latent vectors.
- Insert a 'latent-thought phase' during forward; re-insert as KV before answer decoding.
- Distill from teacher CoT or longer rationale when available.
- Track k, acceptance, and answer F1.
