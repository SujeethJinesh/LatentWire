# COLM_v3 Reviewer Revision References

- date: 2026-05-05
- scope: primary sources added or foregrounded while addressing COLM_v3 reviewer comments

## Distributed/Side-Information Framing

- Slepian and Wolf (1973), "Noiseless Coding of Correlated Information Sources."
  Used only as background motivation for side information at the decoder.
- Wyner and Ziv (1976), "The Rate-Distortion Function for Source Coding with
  Side Information at the Decoder." Used only as background motivation.
- Whang et al. (2021), "Neural Distributed Source Coding,"
  <https://arxiv.org/abs/2106.02797>. Relevant because it is a learned
  distributed-source-coding baseline with explicit encoder/decoder framing.
- Ozyilkan, Ballé, and Erkip (2023), "Learned Wyner-Ziv Compressors Recover
  Binning," <https://arxiv.org/abs/2305.04380>. Relevant because it shows what
  formal learned Wyner-Ziv work looks like; COLM_v3 now explicitly disclaims
  such a formal rate-distortion result.

## Inter-Model / Cache Communication Baselines

- Shi et al. (2025), "KVComm: Enabling Efficient LLM Communication through
  Selective KV Sharing," OpenReview ICLR 2026. Dense/cache-level communication
  baseline, not a byte-scale packet protocol.
- Ye et al. (2025), "KVCOMM: Online Cross-context KV-cache Communication for
  Efficient LLM-based Multi-agent Systems," <https://arxiv.org/abs/2510.12872>.
  Added to disambiguate the concurrent KVComm/KVCOMM naming collision.
- Pham et al. (2024), "Let Models Speak Ciphers: Multiagent Debate through
  Embeddings," <https://openreview.net/forum?id=sehRvaIPQQ>. Relevant as
  embedding-mediated inter-model communication.
- Westphal et al. (2026), "Hide and Seek in Embedding Space,"
  <https://arxiv.org/abs/2601.22818>. Relevant as embedding-space
  steganography; COLM_v3 now avoids unqualified privacy terminology.

## Destructive-Control Precedents

- Hewitt and Liang (2019), "Designing and Interpreting Probes with Control
  Tasks," <https://arxiv.org/abs/1909.03368>.
- Adebayo et al. (2018), "Sanity Checks for Saliency Maps."
- McCoy, Pavlick, and Linzen (2019), "Right for the Wrong Reasons: Diagnosing
  Syntactic Heuristics in Natural Language Inference."
- Ribeiro et al. (2020), "Beyond Accuracy: Behavioral Testing of NLP Models
  with CheckList," <https://aclanthology.org/2020.acl-main.442/>.

## Paper Boundary Changes

- "Source-private" is replaced in the paper claim with "content-private" and
  "source-state-private." The packet leaks source choice with high probability,
  so it is not a formal privacy mechanism.
- "Slepian-Wolf/Wyner-Ziv" is now explicitly inspirational rather than a formal
  theorem.
- "Falsification protocol" novelty is softened to a comprehensive destructive
  control suite drawing on established evaluation precedents.
- Systems rows are now framed as shape-restricted task hints versus
  general-purpose state transfer, avoiding a function-equivalence claim against
  C2C/KV baselines.
