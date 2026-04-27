# references/repos/StreamingLLM/assets/StreamingLLM.pdf

<!-- page 1 -->

Guangxuan Xiao¹, Yuandong Tian², Beidi Chen³, Song Han¹, Mike Lewis²
Efficient Streaming Language Models
with Attention Sinks
Massachusetts Institute of Technology¹
Meta AI²
Carnegie Mellon University³

<!-- page 2 -->

Motivation: Use cases
2


<!-- page 3 -->

Challenges of Deploying LLMs in Streaming Applications
•Urgent need for LLMs in streaming
applications such as multi-round
dialogues, where long interactions are
needed.
3
https://github.com/tomaarsen/attention_sinks
•Challenges:
•Extensive memory consumption during
the decoding stage.
•Inability of popular LLMs to generalize to
longer text sequences.


<!-- page 4 -->

Challenges of Deploying LLMs in Streaming Applications
4


<!-- page 5 -->

Challenges of Deploying LLMs in Streaming Applications
5


<!-- page 6 -->


The Problem of Long Context: Large KV Cache
The KV cache could be large with long context
•During Transformer decoding (GPT-style), we need to store the Keys and Values of all previous
tokens so that we can perform the attention computation, namely the KV cache
•Only need the current query token
6
Image credit: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/transformers-neuronx/generative-llm-inference-with-neuron.html


<!-- page 7 -->


The Problem of Long Context: Large KV Cache
The KV cache could be large with long context
•During Transformer decoding (GPT-style), we need to store the Keys and Values of all previous
tokens so that we can perform the attention computation, namely the KV cache
•Only need the current query token
7
Image credit: https://medium.com/@joaolages/kv-caching-explained-276520203249
7

<!-- page 8 -->


The Problem of Long Context: Large KV Cache
The KV cache could be large with long context
•During Transformer decoding (GPT-style), we need to store the Keys and Values of all previous
tokens so that we can perform the attention computation, namely the KV cache
•Only need the current query token
8
Image credit: https://medium.com/@joaolages/kv-caching-explained-276520203249

<!-- page 9 -->


The Problem of Long Context: Large KV Cache
The KV cache could be large with long context
•We can calculate the memory required to store the KV cache
•Take Llama-2-7B as an example
•Now we calculate the KV cache size under  and diﬀerent sequence lengths.
•Quickly larger than model weights
BS=4
9
BS⏟
batchsize
*32⏟
layers
*32⏟
kv−heads
*128⏟
nemd
*N
⏟
length
*2⏟
K&V
*2bytes
FP16
=0.5MB×BS×N
KV cache size (GB)
2
18
34
50
66
1K 2K 4K 8K 16K 32K
MHA
Model Size
Sequence Length
4*32*32*128*32K*2*2=64GB

<!-- page 10 -->

The Limits of Window Attention
•A natural approach — window attention: caching only the most recent Key-Value states.
•Drawback: model collapses when the text length surpasses the cache size, when the initial token
is evicted.
10
(b) Window Attention
⋯
L cached
tokens
⋯
T-L evicted
tokens
O(TL) PPL: 5158
Breaks when initial tokens
are evicted.

<!-- page 11 -->

Difficulties of Other Methods
11
(a) Dense Attention
⋯
T cached tokens
Current Token
(c) Sliding Window
w/ Re-computation
L re-computed
tokens
⋯
previous tokens are
truncated
O(T2) O(TL2)PPL: 5641 PPL: 5.43
Has poor efficiency and
performance on long text.
(b) Window Attention
⋯
L cached
tokens
⋯
T-L evicted
tokens
O(TL) PPL: 5158
Breaks when initial
tokens are evicted.
Has to re-compute cache
for each incoming token.

<!-- page 12 -->

The “Attention Sink” Phenomenon
•Observation: initial tokens have large attention scores, even if they're not semantically signiﬁcant.
•Attention Sink: Tokens that disproportionately attract attention irrespective of their relevance.
12


<!-- page 13 -->

The “Attention Sink” Phenomenon
•This phenomenon is observed in the SpAtten paper three years ago, but was not explored.
13
SpAtten: Eﬃcient Sparse Attention Architecture with Cascade Token and Head Pruning


<!-- page 14 -->

•Does the importance of the initial tokens
arise from their position or their semantics?
•We found adding initial four “\n”s can also recover perplexity.
•Therefore, it is position!
Understanding Why Attention Sinks Exist
The Rationale Behind Attention Sinks
•SoftMax operation's role in creating attention
sinks — attention scores have to sum up to one
for all contextual tokens.
•Initial tokens' advantage in becoming sinks due
to their visibility to subsequent tokens, rooted
in autoregressive language modeling.
14


<!-- page 15 -->

•Objective: Enable LLMs trained with a ﬁnite attention window to handle inﬁnite text lengths
without additional training.
•Key Idea: preserve the KV of attention sink tokens, along with the sliding window's KV to
stabilize the model's behavior.
StreamingLLM: Using Attention Sinks for Infinite Streams
15
(d) StreamingLLM (ours)
Attention Sink
⋯ L cached
tokens
⋯
evicted
tokens
O(TL) PPL: 5.40
Can perform efficient and stable
language modeling on long texts.
Attention Sinks
01234567Generating
Token 7
012345678Generating
Token 8
0123456789
Evicted Tokens Rolling KV Cache
Generating
Token 9

<!-- page 16 -->

•Use positions in the cache instead of those in the original text.
Positional Encoding Assignment
16
Attention Sinks
0123456789
Evicted Tokens Rolling KV Cache
Generating
Token 9
0123 4567Assigned
Positions

<!-- page 17 -->

Streaming Performance
•Comparison between dense attention, window attention, and sliding window w/ re-computation.
17


<!-- page 18 -->

Streaming Performance
Super Long Language Modeling
•With StreamingLLM, model families include Llama-2, MPT, Falcon, and Pythia can now eﬀectively
model up to 4 million tokens.
18


<!-- page 19 -->

Efficiency
•Comparison baseline: The sliding window with re-computation, a method that is
computationally heavy due to quadratic attention computation within its window.
•StreamingLLM provides up to 22.2x speedup over the baseline, making LLMs for real-time
streaming applications feasible.
19


<!-- page 20 -->

Ablation Study: #Attention Sinks
•The number of attention sinks that need to be introduced to recover perplexity.
•4 attention sinks are generally enough.
20


<!-- page 21 -->

Pre-training with a Dedicated Attention Sink Token
•Idea: Why 4 attention sinks? Can we train a LLM that need only one single attention sink? Yes!
•Method: Introduce an extra learnable token at the start of all training samples to act as a
dedicated attention sink.
•Result: This pre-trained model retains performance in streaming cases with just this single sink
token, contrasting with vanilla models that require multiple initial tokens.
21


<!-- page 22 -->

Thanks for Listening!
•We propose StreamingLLM, enabling the streaming deployment of LLMs.
•Paper: https://arxiv.org/abs/2309.17453
•Code:  https://github.com/mit-han-lab/streaming-llm
•Demo: https://youtu.be/UgDcZ3rvRPg
22
