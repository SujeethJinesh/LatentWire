# references/repos/Quest/assets/quest_poster.pdf

<!-- page 1 -->

Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
Jiaming Tang1,2*, Yilong Zhao1,3*, Kan Zhu3, Guangxuan Xiao2, Baris Kasikci3, Song Han2,4
1SJTU, 2MIT, 3UW, 4NVIDIA        https://github.com/mit-han-lab/Quest
Introduction
•Long-context text-generation has gained popular applications.
•However, it poses great memory pressure to the inference
system.
•In this work, we prose Quest, which exploits query-aware
sparsity in self-attention operator to boost inference efficiency,
with negligible accuracy loss.
Attention in Decode Phase is Costly
•Decode phase consumes great portion of time compared to
prefill phase, due to the auto-regressive inference of LLMs.
Finding 1: Attention is Sparse
•During attention calculation, only small portion of tokens
has much larger magnitude of attention scores than
others.
•Therefore, attention output can be effectively
approximated by only a small portion of critical tokens,
which is called sparsity.
Finding 2: Sparsity Depends on Query
•We argue that critical tokens depend on the input
query. For e.g., summary task will attend on different
paragraphs, sequentially.
•Therefore, query-agnostic sparsity (like H2O) will prune
tokens which will be critical in future.
Overview of Quest
•Instead of prune, Quest dynamically selects critical tokens via
criticality estimation for each query, at the page granularity which is
compatible with PageAttention.
•Quest applies sparse attention only on the selected tokens, which
saves greatly memory movement.
Evaluation & Implementation
•We implemented specialized operators (criticality estimation, Top-
K, sparse attention), based on kernel libraries, RAFT and FlashInfer.
•For efficiency evaluation, we run experiments on Ada 6000 with
CUDA 12.3.
•For accuracy evaluation, we evaluate on Pass-Key retrieval and
common sense tasks from LongBench.
LongBench Tasks
•Quest consistently outperforms all baselines.
•Note that Quest achieves full accuracy with 2K budgets in most cases.
(a) Dense Attention
Current
Token
(c) Query-Aware
Sparsity (ours)
Keeps all
contextual tokens.
(b) Query-Agnostic Sparsity
 (StreamingLLM, , etc.)H2O
O(L) Acc: 2%
O(T) Acc: 100%
 Acc: 100%
O(L)
Once a token is
evicted, it cannot be
attended anymore.
Keeps tokens based on
past information.
Keeps tokens based
on current tokens.
An evicted token
can still be attended
by future tokens.
Token ‘B’ was not critical
Token ‘B’ is critical now
Keys ⋯
Page 1 Page 2 Page N
KV Cache
(Global Memory)
Element-wise Min Key
Element-wise Max Key
-1-4-1-1-1-4-4-4
4 3 4 3 3 2 1 3
-4-3-3-3-32-3-2
2 2 3 3 3 3 0 4
Reduced Keys
-11 3 0 1 0-32
 Current Query
(SRAM)
Element-wise
Product
Per-channel
Max
80 32
Top-K
4-34-13-41 3
-3-20 3 0-4-2-3
-13-13 1-4-4-3
-2-40 3-12-3-4
-40 3 3 3 3-23
2 2-3-3-33-2-2
-3-30 2 0 2 0 4
-2-1-1-32 2-3-1
⋯
1-4-30-1012-8
-43120 3 0-36
1-3-90-30 9-4
-22 9 0 3 0 0 8
⊙
⋯
⊙ ⊙
1 3120 3 0126 1 2 9 0 3 0 9 8
Sum 37
-40 3 3 3 3-23
2 2-3-3-33-2-2
-3-30 2 0 2 0 4
-2-1-1-32 2-3-1
4-34-13-41 3
-3-20 3 0-4-2-3
-13-13 1-4-4-3
-2-40 3-12-3-4
Stage 2: Compute Sparse Attention
Stage 1: Estimate Critical Pages
⋯
Load Load Load
Pass-Key Retrieval
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
QUEST: Query-Aware Sparsity for Efﬁcient Long-Context LLM Inference
Algorithm 1 Token Criticality Estimation
When inserting new token to KV cache:
Input: Key vector K, Dimension of hidden states dim ,
Current maximal vector Mi, Current minimal vector mi
for i=1 todim do
Mi=max( Mi,ki)
mi=min( mi,ki)
end for
When perform self-attention:
Input: Query vector Q, Dimension of hidden states dim ,
Current maximal vector Mi, Current minimal vector mi
Initialize score =0 .
for i=1 todim do
score += M AX (qi⇤max, q i⇤min )
end for
3.5. Quest Reduces the Memory Movement of
Self-Attention
Instead of loading the whole KV cache, Quest only needs
to load a fraction of the data, which leverages query-aware
sparsity. Assume that every K or V vector is M bytes,
the KV cache contains Ltokens, and each page contains S
KV pairs (Page size). During criticality estimation, Quest
will load maximal and minimal vectors of each page, which
is approximately 2M ⇤L/S bytes. Additionally, Quest
performs normal self-attention for top K pages, which is
2M ⇤K⇤Sbytes. The whole KV cache is 2M ⇤Lbytes,
which indicates Quest loads 1/S+K⇤S/L of the total KV
cache ‡
, which is equivalent to
1
Page Size
+
K
Page Num
Assuming that we use 16KV pairs per page, context length
is 64K, and we choose the top 4K pages, Quest will reduce
the memory load by 8⇥. Note that this memory load reduc-
tion is universal across all models and is compatible with
existing quantization mechanisms ( Zhao et al. ,2023 ).
4. Experiments
4.1. Setting
We evaluate Quest on the language modeling dataset
PG19 ( Rae et al. ,2019 ), passkey retrieval task ( Peng et al. ,
2023 ), and six datasets in LongBench ( Bai et al. ,2023 ):
NarrativeQA ( Koˇcisk ´y et al. ,2018 ), HotpotQA ( Y ang et al. ,
‡
The top-K operator incurs negligible memory loading and exe-
cution time (5-10 us). Therefore, we do not include it in efﬁciency
analysis.
Method / Budget 32 64 128 256 512
H2O 0% 1% 1% 1% 3%
TOV A 0% 1% 1% 3% 8%
StreamingLLM 1% 1% 1% 3% 5%
Quest (ours) 65% 99% 99% 99% 100%
Method / Budget 256 512 1024 2048 4096
H2O 2% 2% 2% 2% 4%
TOV A 2% 2% 2% 2% 10%
StreamingLLM 1% 1% 1% 2% 4%
Quest (ours) 88% 92% 96% 100% 100%
Table 1. (i) Results of 10k length passkey retrieval test on
LongChat-7b-v1.5-32k. (ii) Results of 100k length passkey re-
trieval test on Y arn-Llama-2-7b-128k. Quest can achieve nearly
perfect accuracy with a KV cache of 64 and 1024 tokens, which
is about 1% of the total sequence length, demonstrating that
Quest can effectively preserve the model’s ability to handle long-
dependency tasks. However, KV cache eviction algorithms such
as H2O, TOV A, and StreamingLLM incorrectly discard the KV
cache of the answer before receiving the question, thus failing to
achieve ideal accuracy.
2018 ), Qasper ( Dasigi et al. ,2021 ), TrivialQA ( Joshi et al. ,
2017 ), GovReport ( Huang et al. ,2021 ), MultiﬁeldQA ( Bai
et al. ,2023 ). We choose two widely used long-context
models for our evaluation: LongChat-v1.5-7b-32k ( Li et al. ,
2023 ) and Y arn-Llama-2-7b-128k ( Peng et al. ,2023 ). We
compare our method against the cache eviction algorithm
H2O ( Zhang et al. ,2023b ), TOV A ( Oren et al. ,2024 ), and
the sliding window method with StreamingLLM ( Xiao et al. ,
2023 ). Note that we do not apply any Quest and other base-
line algorithms to the ﬁrst two layers of the model, as our
analysis in Sec 3.4 indicates a low sparsity ratio for these
layers.
4.2. Accuracy Evaluation
4.2.1. L ANGUAGE MODELING ON PG19
We ﬁrst evaluate the language modeling perplexity on the
PG19 test set, which is a dataset comprising 100 books with
an average length of 70k tokens. We use the LongChat-7b-
v1.5-32k model to test 32k tokens on PG19. We feed the
model with various numbers of tokens and evaluate the per-
plexity of generated tokens. We evaluate H2O, TOV A, and
Quest with a token budget of 4096, which is approximately
1/8 of the total token length. As indicated by the perplexity
results in Fig. 4, Quest’s accuracy closely matches the oracle
baseline with a full KV cache.
4.2.2. R ESULTS ON LONG TEXT PASSKEY RETRIEV AL
TASK
Since language modeling evaluation only involves local
dependencies, models can achieve great performance by
5
LongChat- 7b-v1.5-32k with various token budgets
End-to-end Efficiency
•Breakdown of Quest’s attention operator under various context length.
1024
2048
4096
Full Cache
Latency (ms)
0 40 80 120 160
Criticality EstimationTop-K FilteringApproximate Attention
Sequence Length: 8192
Latency (ms)
0 64128192256320
Sequence Length: 16384
Latency (ms)
0 150 300 450 600
Sequence Length: 30720
1.69x 2.92x 4.62x
Latency (ms)
0
10
20
30
40
Context Length
8192 16384 30720
21.5
20.4
19.8
20.2
19.1
18.5
19.6
18.5
17.9
19.2
18.2
17.6
35.7
26.8
21.8
FlashInfer512 1024 2048 4096
(a) FP16 Weight
Latency (ms)
0
7.5
15
22.5
30
Context Length
8192 16384 30720
13.7
12.9
12.4
12.4
11.6
11.1
11.8
11.0
10.5
11.4
10.6
10.2
28.1
19.4
14.4
(b) 4-bit Weight (AWQ)
•End-to-end speedup compared to FlashInfer version.
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
QUEST: Query-Aware Sparsity for Efﬁcient Long-Context LLM Inference
Algorithm 1 Token Criticality Estimation
When inserting new token to KV cache:
Input: Key vector K, Dimension of hidden states dim ,
Current maximal vector Mi, Current minimal vector mi
for i=1 todim do
Mi=max( Mi,ki)
mi=min( mi,ki)
end for
When perform self-attention:
Input: Query vector Q, Dimension of hidden states dim ,
Current maximal vector Mi, Current minimal vector mi
Initialize score =0 .
for i=1 todim do
score += M AX (qi⇤max, q i⇤min )
end for
3.5. Quest Reduces the Memory Movement of
Self-Attention
Instead of loading the whole KV cache, Quest only needs
to load a fraction of the data, which leverages query-aware
sparsity. Assume that every K or V vector is M bytes,
the KV cache contains Ltokens, and each page contains S
KV pairs (Page size). During criticality estimation, Quest
will load maximal and minimal vectors of each page, which
is approximately 2M ⇤L/S bytes. Additionally, Quest
performs normal self-attention for top K pages, which is
2M ⇤K⇤Sbytes. The whole KV cache is 2M ⇤Lbytes,
which indicates Quest loads 1/S+K⇤S/L of the total KV
cache ‡
, which is equivalent to
1
Page Size
+
K
Page Num
Assuming that we use 16KV pairs per page, context length
is 64K, and we choose the top 4K pages, Quest will reduce
the memory load by 8⇥. Note that this memory load reduc-
tion is universal across all models and is compatible with
existing quantization mechanisms ( Zhao et al. ,2023 ).
4. Experiments
4.1. Setting
We evaluate Quest on the language modeling dataset
PG19 ( Rae et al. ,2019 ), passkey retrieval task ( Peng et al. ,
2023 ), and six datasets in LongBench ( Bai et al. ,2023 ):
NarrativeQA ( Koˇcisk ´y et al. ,2018 ), HotpotQA ( Y ang et al. ,
‡
The top-K operator incurs negligible memory loading and exe-
cution time (5-10 us). Therefore, we do not include it in efﬁciency
analysis.
Method / Budget 32 64 128 256 512
H2O 0% 1% 1% 1% 3%
TOV A 0% 1% 1% 3% 8%
StreamingLLM 1% 1% 1% 3% 5%
Quest (ours) 65% 99% 99% 99% 100%
Method / Budget 256 512 1024 2048 4096
H2O 2% 2% 2% 2% 4%
TOV A 2% 2% 2% 2% 10%
StreamingLLM 1% 1% 1% 2% 4%
Quest (ours) 88% 92% 96% 100% 100%
Table 1. (i) Results of 10k length passkey retrieval test on
LongChat-7b-v1.5-32k. (ii) Results of 100k length passkey re-
trieval test on Y arn-Llama-2-7b-128k. Quest can achieve nearly
perfect accuracy with a KV cache of 64 and 1024 tokens, which
is about 1% of the total sequence length, demonstrating that
Quest can effectively preserve the model’s ability to handle long-
dependency tasks. However, KV cache eviction algorithms such
as H2O, TOV A, and StreamingLLM incorrectly discard the KV
cache of the answer before receiving the question, thus failing to
achieve ideal accuracy.
2018 ), Qasper ( Dasigi et al. ,2021 ), TrivialQA ( Joshi et al. ,
2017 ), GovReport ( Huang et al. ,2021 ), MultiﬁeldQA ( Bai
et al. ,2023 ). We choose two widely used long-context
models for our evaluation: LongChat-v1.5-7b-32k ( Li et al. ,
2023 ) and Y arn-Llama-2-7b-128k ( Peng et al. ,2023 ). We
compare our method against the cache eviction algorithm
H2O ( Zhang et al. ,2023b ), TOV A ( Oren et al. ,2024 ), and
the sliding window method with StreamingLLM ( Xiao et al. ,
2023 ). Note that we do not apply any Quest and other base-
line algorithms to the ﬁrst two layers of the model, as our
analysis in Sec 3.4 indicates a low sparsity ratio for these
layers.
4.2. Accuracy Evaluation
4.2.1. L ANGUAGE MODELING ON PG19
We ﬁrst evaluate the language modeling perplexity on the
PG19 test set, which is a dataset comprising 100 books with
an average length of 70k tokens. We use the LongChat-7b-
v1.5-32k model to test 32k tokens on PG19. We feed the
model with various numbers of tokens and evaluate the per-
plexity of generated tokens. We evaluate H2O, TOV A, and
Quest with a token budget of 4096, which is approximately
1/8 of the total token length. As indicated by the perplexity
results in Fig. 4, Quest’s accuracy closely matches the oracle
baseline with a full KV cache.
4.2.2. R ESULTS ON LONG TEXT PASSKEY RETRIEV AL
TASK
Since language modeling evaluation only involves local
dependencies, models can achieve great performance by
5
•Attention operator needs read entire KV-Cache at each iteration,
which increases linearly with the context length.
Normalized Latency
00.10.20.30.40.50.60.70.80.91
Decode : Preﬁll
1/641/321/161/81/41/2
Preﬁll Decode
5.9%3.1%1.5%0.8%0.4%0.2%
Normalized Latency
00.10.20.30.40.50.60.70.80.91
Context Length
4096819212288163842048024576
OthersAttention
Normalized Latency
00.10.20.30.40.50.60.70.80.91
Decode : Preﬁll
1/641/321/161/81/41/2
Preﬁll Decode
5.9%3.1%1.5%0.8%0.4%0.2%
Normalized Latency
00.10.20.30.40.50.60.70.80.91
Context Length
4096819212288163842048024576
OthersAttention
Softmax
Less than 1% values
•10K context length tested on LongChat-7b-v1.5-32k
•100K context length tested on Yarn-Llama2-7b-128k
Sparsity of maintaining PG19 Perplexity
Sampled Attention Map Top-10 Recall Rate with Various
Token Budgets (10K Context Len)
Inference Time Breakdown
Time ratio under 1K prefill length with various decode length
64
256
1024
0 0.25 0.5 0.75 1
Full Quest H2O
