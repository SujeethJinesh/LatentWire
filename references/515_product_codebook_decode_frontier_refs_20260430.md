# Product-Codebook Decode Frontier References

Date: 2026-04-30

Blocker addressed: the product-codebook packet method functionally passed but
needed a defensible systems story separating source packet construction,
request-level public table construction, and receiver lookup latency.

## Sources

1. Product Quantization for Nearest Neighbor Search. IEEE TPAMI, 2011.
   DOI:10.1109/TPAMI.2010.57. https://doi.org/10.1109/TPAMI.2010.57
   - Mechanism: split vectors into subspaces, encode each subspace with a
     compact learned centroid index, and search with compact codes.
   - Experiment implication: the packet interface should report one centroid
     index per byte and compare against target-side candidate vectors.
   - Role: baseline and theory support.

2. Optimized Product Quantization. CVPR 2013.
   https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
   - Mechanism: rotate vector space before product quantization to reduce
     distortion.
   - Experiment implication: a future packet variant should test OPQ-style
     rotation before codebook fitting; current result is the non-rotated PQ
     baseline.
   - Role: ablation and inspiration.

3. Faiss implementation notes: product quantization and distance tables.
   https://github.com/facebookresearch/faiss/wiki/Implementation-notes
   - Mechanism: PQ search commonly precomputes query/product-codebook distance
     tables and then scores compact codes by table lookup.
   - Experiment implication: receiver-side decode should include a table-lookup
     row, not only full-vector reconstruction.
   - Role: systems implementation support.

4. Quicker ADC: Unlocking the Hidden Potential of Product Quantization with SIMD.
   IEEE TPAMI, 2019. https://arxiv.org/abs/1812.09162
   - Mechanism: accelerate asymmetric distance computation over PQ codes with
     SIMD-friendly lookup and accumulation.
   - Experiment implication: native kernels should be expected to improve over
     the current Python/NumPy table path; report resident lookup separately from
     request table construction.
   - Role: systems inspiration and future baseline.

5. Nearest Neighbor Search with Compact Codes: A Decoder Perspective.
   arXiv:2112.09568. https://arxiv.org/abs/2112.09568
   - Mechanism: evaluates compact-code methods through the decoder/search side,
     not only encoder distortion.
   - Experiment implication: add decoder-side latency and parity checks to the
     product-codebook packet result.
   - Role: framing and ablation support.

6. Neural Discrete Representation Learning. NeurIPS 2017.
   https://arxiv.org/abs/1711.00937
   - Mechanism: learned discrete codebooks as neural latent bottlenecks.
   - Experiment implication: future learned packet variants should log codebook
     usage/perplexity and compare k-means PQ against trained VQ codebooks.
   - Role: inspiration.

7. Python `pyperf` documentation.
   https://pyperf.readthedocs.io/en/latest/run_benchmark.html
   - Mechanism: structured Python benchmark warmup/repetition discipline.
   - Experiment implication: the current frontier is enough for a project gate,
     but paper-grade timing should be repeated under `pyperf` or equivalent
     environment controls.
   - Role: benchmark methodology.

8. Google Benchmark user guide.
   https://github.com/google/benchmark/blob/main/docs/user_guide.md
   - Mechanism: warns about CPU scaling and benchmark environment effects.
   - Experiment implication: final systems claims need environment metadata and,
     ideally, a native microbenchmark once NVIDIA/GPU hardware is available.
   - Role: benchmark methodology.

## Effect On This Cycle

The next experiment changed from “make the product-codebook decoder faster” to
“measure the correct decode contract.” The resulting frontier reports source
packet construction, source packet kernel, cold receiver decode, request-public
table decode, resident table lookup, and batch amortization separately. This
turns the previous latency failure into a scoped systems-positive finding while
preserving the caveat that full serving throughput is not yet measured.
