# references/repos/Quest/assets/quest_slides.pdf

<!-- page 1 -->

Quest: Query-Aware Sparsity for
Efficient Long-Context LLM Inference
Jiaming Tang*, Yilong Zhao*, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han
Massachusetts Institute of Technology
University of Washington
Shanghai Jiao Tong University
NVIDIA
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 2 -->

MotivationWhy Long Context LLMs?
2
8,000 tokens 128,000 tokens 1,000,000 tokens
A short blog A Harry Potter book The whole series of Harry Potter books
What inputs can LLMs handle with different context lengths?
Context
Length
A short video vlog
 30 seconds~
A short film – Piper
 10 minutes~
A film
1 - 2 hours
What videos can multi-modal LLMs process (given a rate of 200 tokens per frame per second)?
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 3 -->

Motivation
•As the demand for long-context large language models (LLMs) increases, models with context
windows of up to 128k or even 1M tokens are becoming increasingly prevalent.
3
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 4 -->

The Problem of Long Context: Large KV Cache
Large KV cache slows down long context inference
•However, long-context LLM inference is challenging since the inference speed decreases
signiﬁcantly as the sequence length grows.
•This slowdown is primarily caused by loading a large KV cache during attention.
4QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)


<!-- page 5 -->

The Sparsity in Attention Mechanism
5
•Previous research has highlighted the inherent sparsity in attention mechanism.
•Due to this property of self-attention, a small portion of tokens in the KV cache, called critical
tokens, can accumulate suﬃcient attention scores, capturing the most important inter-token
relationships.
H2O: Heavy-Hitter Oracle for Eﬃcient Generative Inference of Large Language Models. Zhang et al.
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)


<!-- page 6 -->

The Limits of Previous Methods
•Many previous eﬀorts have been dedicated to compressing the size of the KV cache to
accelerate attention and reduce memory usage.
•These methods decide which parts of the KV cache to discard based on historical information or
current states, but discarded tokens might be important for future tokens, which may cause
the loss of important information.
6
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 7 -->

The Limits of Previous Methods
•The criticality of the tokens is dynamic and highly dependent on the query vector Q.
•Example: the token ‘B’ is critical to the current query ‘is’. Thus, it has a high attention score.
However, before the ﬁnal token ‘is’, ‘B’ is not critical for any previous query and has very low
attention scores.
7
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 8 -->

•Key Idea: preserve all KV cache, and signiﬁcantly accelerate inference by reducing the memory
movement from the entire KV cache to selected constant K pages.
Quest: Using Query-aware Sparsity in Attention
8
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 9 -->

•Our insight is that in order not to miss critical tokens, we should select pages containing the
tokens with the highest attention weights.
•However, for an eﬃcient selection of pages, we should calculate an approximate attention
score following this insight.
•We found that the upper bound attention weights within a page can be used to approximate
the highest attention score in the page.
Quest: Using Query-aware Sparsity in Attention
9
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 10 -->

Quest Performance
Needle-in-a-Haystack
•(i) Results of 10k length passkey retrieval test on LongChat-7b-v1.5-32k.
•(ii) Results of 100k length passkey retrieval test on Yarn-Llama-2-7b-128k.
•Quest can achieve nearly perfect accuracy with a KV cache of 64 and 1024 tokens, which is
about 1% of the total sequence length, demonstrating that Quest can eﬀectively preserve the
model’s ability to handle long-dependency tasks.
10
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 11 -->

Quest Performance
Super Long Language Modeling
•Language modeling evaluation of Quest on PG19 dataset.
•Quest can closely match the performance of the full cache model.
11
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 12 -->

Quest Performance
LongBench
•We evaluate LongChat-7b-v1.5-32k across a wide range of long-context datasets,
•Quest with a budget of 2k tokens can achieve comparable performance as the model with
full KV cache, while other baselines still exhibit a notable gap from full cache performance even
with a larger budget.
•Single-document QA: NarrativeQA, Qasper, MultiFieldQA; multi-document QA: HotpotQA;
summarization: GovReport; few-shot learning: TriviaQA.
12QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 13 -->

Quest Performance
LongBench
•We evaluate LongChat-7b-v1.5-32k across a wide range of long-context datasets,
•Quest with a budget of 2k tokens can achieve comparable performance as the model with
full KV cache, while other baselines still exhibit a notable gap from full cache performance even
with a larger budget.
•Single-document QA: NarrativeQA, Qasper, MultiFieldQA; multi-document QA: HotpotQA;
summarization: GovReport; few-shot learning: TriviaQA.
13QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 14 -->

Efficiency Evaluation
Quest attention time breakdown
•At all sequence lengths, Quest signiﬁcantly outperforms FlashInfer, as the memory movement
is reduced.
•Quest speeds up self-attention by 7.03× at sequence length 32k with token budget 2048.
14
1024
2048
4096
Full Cache
Latency (us)
0 40 80 120 160
Criticality Estimation Top-K Filtering Approximate Attention
Sequence Length: 8192
Latency (us)
0 64 128 192 256 320
Sequence Length: 16384
Latency (us)
0 163 325 488 650
Sequence Length: 32768
1.69x 2.92x 4.86x
1024
2048
4096
Full Cache
Latency (ms)
0 40 80 120 160
Sequence Length: 8192
0
0.2
0.4
0.6
0.8
1
1/64 1/32 1/16 1/8 1/4 1/2
QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)

<!-- page 15 -->

Efficiency Evaluation
End-to-end latency
•For all sequence lengths, Quest signiﬁcantly outperforms FlashInfer. Increasing the sequence
lengths only slightly changes the latency of Quest.
•Quest speedup end-to-end inference by 2.23× with sequence length 30K, token budget 2048,
4-bit weight quantization.
15QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)
Latency (ms)
0
10
20
30
40
Context Length
8192 16384 32768
22.4
20.7
19.8
21.2
19.4
18.6
20.7
18.9
18.1
20.3
18.5
17.7
36.8
26.6
21.6
FlashInfer 512 1024 2048 4096
(a) FP16 Weight
Latency (ms)
0
7.5
15
22.5
30
Context Length
8192 16384 32768
14.6
12.9
12.4
13.3
11.6
11.1
12.8
11.0
10.5
12.4
10.6
10.2
29.6
19.4
14.4
(b) 4-bit Weight (AWQ)
2.23x1.74x

<!-- page 16 -->

Efficiency Evaluation
Eﬃciency comparison with baselines
•Baselines need nearly full cache to achieve lossless performance on LongBench benchmarks.
•Therefore, Quest outperforms the baseline by up to 4.54x with the same lossless accuracy.
16QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference (ICML 2024)
Context Length
0
8333
16667
25000
Qasper HotpotQA GovReport TriviaQA NarrativeQA MultiFieldQA
1024
5120
512
1024
1024
512
8192
14336
10240
10240
5120
5120
8154
24723
14101
11984
15370
5819
Full TOVA Quest
(a) Average context length for lossless accuracy of attention mechanisms
Attention latency (us)0
167
333
500
Qasper HotpotQA GovReport TriviaQA NarrativeQA MultiFieldQA
64.6
187.2
60.4
71.3
77.7
43.2
222.0
357.5
274.3
272.4
143.1
141.9
222.2
551.2
352.8
315.0
377.1
161.8
(b) Inference latency of different attention mechanism under benchmarks
3.8x 4.5x

<!-- page 17 -->

Thanks for Listening!
•We propose Quest, an eﬃcient long-context LLM inference framework that
leverages query-aware sparsity in the KV cache to accelerate the attention mechanism.
•Code: https://github.com/mit-han-lab/Quest
•Paper: https://github.com/mit-han-lab/Quest/blob/main/assets/quest_paper.pdf
•Poster: https://github.com/mit-han-lab/Quest/blob/main/assets/quest_poster.pdf
17
