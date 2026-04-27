# references/131_query_focused_retrieval_heads_improve_long_context_reasoning_and_re_ranking.pdf

<!-- page 1 -->

Query-Focused Retrieval Heads Improve
Long-Context Reasoning and Re-ranking
Wuwei Zhang♠ Fangcong Yin♢ Howard Y en♠ Danqi Chen♠ Xi Y e♠
♠ Princeton Language and Intelligence, Princeton University
♢ The University of Texas at Austin
♠ {wuwei.zhang,hyen,danqic}@cs.princeton.eduxi.ye@princeton.edu
♢ fangcongyin@utexas.edu
Abstract
Recent work has identified retrieval heads (Wu
et al., 2025b), a subset of attention heads re-
sponsible for retrieving salient information in
long-context language models (LMs), as mea-
sured by their copy-paste behavior in Needle-
in-a-Haystack tasks. In this paper, we in-
troduce QRHEAD( Query-Focused Retrieval
Head), an improved set of attention heads that
enhance retrieval from long context. We iden-
tify QRHEADby aggregating attention scores
with respect to the input query, using a hand-
ful of examples from real-world tasks (e.g.,
long-context QA). We further introduce QR-
RETRIEVER, an efficient and effective retriever
that uses the accumulated attention mass of
QRHEADas retrieval scores. We use QR-
RETRIEVERfor long-context reasoning by se-
lecting the most relevant parts with the high-
est retrieval scores. On multi-hop reasoning
tasks LongMemEval and CLIPPER, this yields
over 10% performance gains over full context
and outperforms strong dense retrievers. We
also evaluate QRRETRIEVERas a re-ranker on
the BEIR benchmark and find that it achieves
strong zero-shot performance, outperforming
other LLM-based re-rankers such as RankGPT.
Further analysis shows that both the query-
context attention scoring and task selection are
crucial for identifying QRHEADwith strong
downstream utility. Overall, our work con-
tributes a general-purpose retriever and offers
interpretability insights into the long-context
capabilities of LMs.1
1 Introduction
Retrieving salient information from long context
serves as a foundation for language models (LMs),
enabling a wide range of downstream applications,
such as long document understanding and passage
re-ranking. Prior work has identified a subset of at-
tention heads in transformers (Vaswani et al., 2017)
1Code:https://github.com/princeton-pli/QRHead.
1K 50K100K
0.1
0.5
0.9
Original Retrieval Heads (Wu et al., 2025)
QRHeads(Ours)0.1
0.5
0.9
Context Length
Depth
Figure 1:Top: Masking the top 32 original retrieval
heads (Wu et al., 2025b) of Llama-3.1-8B.Bottom:
Masking the top 32 QRHeads of the same model,
which has a more pronounced impact on Needle-in-
a-Haystack.
that are responsible for retrieving relevant informa-
tion, known asretrieval heads(Wu et al., 2025b).
However, these retrieval heads are identified
based on the frequency of their copy-paste op-
erations in a simple synthetic task—Needle-in-a-
Haystack (NIAH; Kamradt, 2024). Although they
exhibit significance on certain downstream tasks,
such as extractive question answering, we argue
that the copy-paste objective and synthetic data
used to identify them are misaligned with how lan-
guage models retrieve pertinent information in real-
world settings.
To this end, we propose a more effective ap-
proach for identifying retrieval heads and intro-
duce QRHEAD, a distinct subset of attention heads
whose attention mass plays a more critical role in
retrieving relevant information from long context.
Compared to original retrieval heads, our method
incorporates two key changes: (1) a query-context
scoring function that measures attention mass allo-
cated to pertinent context spans with respect to an
arXiv:2506.09944v2  [cs.CL]  28 Sep 2025

<!-- page 2 -->

input query, and (2) the use of more natural data
from real-world tasks, such as question answering
over long texts. Our method only requires a small
amount of data to be effective. As shown in Fig-
ure 1, we detect QRHEADusing 70 examples from
a natural long-context QA task, LongMemEval,
and find masking them out results in more severe
degradation in NIAH compared to original retrieval
heads detected from in-domain data.
Furthermore, we build QRRETRIEVERon top
of QRHEADas a general-purpose retriever for im-
proving LMs on diverse long-context downstream
applications. Given a query and a set of passages
(e.g., a claim and a book consisting of multiple
chapters), QRRETRIEVERscores each passage us-
ing the aggregated attention mass from the QR-
HEADof a language model, and returns the top-
ranked passages. We detect QRHEADfor multi-
ple LMs of different scales (3B–70B) and families
(Llama-3.1, Llama-3.2, and Qwen), and build QR-
RETRIEVERwith these LLMs.
We evaluate QRRETRIEVERon two long-
context, multi-hop reasoning tasks: Long-
MemEval (Wu et al., 2025a) and CLIPPER(Pham
et al., 2025). Using QRRETRIEVERto select
top-ranked documents yields substantial improve-
ments in retrieval recall and downstream task
performance. For example, with Llama-3.1-8B-
Instruct, QRRETRIEVERoutperforms dense re-
trievers and improves performance by over 10% on
both datasets, compared to full-context generation.
We further evaluate QRRETRIEVERas a re-ranker
on the standard BEIR benchmark (Thakur et al.,
2021). It exhibits strong zero-shot performance
across diverse domains and outperforms other
LLM-based re-rankers, such as RankGPT (Sun
et al., 2024).
Finally, we provide extensive analyses of the ef-
fectiveness of QRHEAD. First, using QRHEAD
outperforms both full attention heads and original
retrieval heads. Second, QRHEADgeneralizes
across different tasks and input lengths—the
heads identified at 32K tokens transfer well to tasks
with 128K context lengths. Lastly, we show that
both key modifications—our query-focused scoring
objective and the use of natural data—contribute
to the improved downstream performance of QR-
HEADover original retrieval heads. Together, these
findings highlight the practicality and robustness of
QRHEADas a foundation for long-context retrieval
and suggest opportunities for further exploration of
retrieval mechanisms in language models.
2 Background: Retrieval Heads
Retrieval heads are a specialized subset of atten-
tion heads that are pivotal for extracting relevant
information from long input context.
Original retrieval heads.Wu et al. (2025b) first
discovered a set of retrieval heads that exhibit
copy-paste behavior during decoding—effectively
copying tokens from the long context input con-
text into the generated output. As shown in Fig-
ure 2 (top), the retrieval head detection method
roots from the Needle-in-a-Haystack test (NIAH)
with a triple (C, q, a) of context, question, and an-
swer: the answer span a (the “needle”) is embedded
within a long context sequence C=d 1...a...dN
where d1, ..., dN are N irrelevant passages (the
“haystack”). The LM is tasked to generate an an-
swer to q based on the provided context. Successful
generation of a demonstrates effective copy-paste
behavior by extracting a from the haystack and
copying it over to the output. We say an attention
head h copies a token t if, during the generation of
t in the answer, h assigns the maximum attention
score to the same token t in the needles. To quan-
tify this behavior, the retrieval score of an attention
head h is defined as the fraction of tokens copied
fromaby the headhduring decoding:
Retrieval_Score(h) = |gh ∩a|
|a| ,(1)
whereg h denotes the set of tokens copied by head
h to the output. Attention heads with the highest
retrieval scores are selected as retrieval heads.
Shortcomings.The scoring mechanism de-
scribed above focuses only on attention heads that
perform strict copy-paste operations, potentially
missing heads involved in semantic-based retrieval,
such as paraphrasing or reasoning over relevant
context. Moreover, recent work has shown that
heads identified through copy-paste metrics exhibit
limited cross-domain generalizability (Zhao et al.,
2025). This suggests that the simplified formula-
tion may not fully capture the complexity of in-
context retrieval behavior in LLMs and has limited
relevance for downstream applications.
3 QRHEAD: Identifying Query-Focused
Retrieval Heads
In this section, we introduce a new approach for de-
tecting retrieval heads that significantly improves
upon prior retrieval head detection. For clarity,

<!-- page 3 -->

R etrie v al Heads (Wu et al., 2025)
QRHeads (Ours)
Once upon   a   time      ... ...     happily e v er aft er .
Irr ele v ant cont e xt Irr ele v ant cont e xt
The best t hing ... eat a sandwich.
Needle sent ence
Q: What is t he best t hing
t o do in San F rancisco? A: t o eat
Heads t hat assign max att ention
scor es t o cop y-past ed t ok ens
During t he 1930s and 1940s ... In e v olutionar y biology , ...
Distr acting cont e xt Distr acting cont e xt
1858 – Charles R. Dar win and ...
Gold cont e xt
Q: who pr oposed e v olution as t he
basis of biological de v elopment?
Heads t hat allocat e att ention mass t o
gold cont e xt giv en a q uer y
Data :  synt hetic ( i.e., N IAH )
Ob j ectiv e :  cop y-pasting
Data :  r ealistic w /  distracting cont e x t
Ob j ectiv e :  q uer y-cont e xt att ention
Distracting cont e xt : Har d negativ es in open-domain
Q A or ot her t e xt s wit hin t he same document
Figure 2: Comparison between Retrieval Heads (Wu et al., 2025b) and QRHEAD(Ours).
we refer to our heads as Query-Focused Retrieval
Head (QRHEAD) and the original retrieval head as
RETHEAD. See Figure 2 (bottom), our approach in-
troduces two key improvements. First, we propose
a query-focused retrieval score (QRscore), which
captures query-context attention rather than rely-
ing solely on copy-paste behavior (§3.1). Second,
we leverage realistic tasks that require in-context
retrieval to identify effective heads (§3.2). We also
present a comparison between QRHEADand RET-
HEAD(§3.3).
Task formulation: LMs for in-context retrieval.
Our study focuses on the task of in-context retrieval
with LMs, i.e., identifying relevant information
from given context. Formally, let Σ denote the vo-
cabulary. Given an input query q∈Σ ∗ and a con-
text D∈Σ ∗, the objective is to retrieve the most
relevant information from the context with respect
to q, denoted as D[q] ⊆D . Typically, the context
D consists of a sequence of passages (or chunks),
represented as D={d 1, d2, . . . , dN }. With both q
and D jointly fed into an LM as input, we assign
a score R(q, di) to each passage di with respect to
q. We measure the effectiveness of the retriever by
evaluating whether the top-scored passages align
with the ground-truth relevant documentsD ∗
[q].2
3.1 Scoring Heads with Query-Context
Attention
Instead of scoring attention heads based on their
activations in copy-paste operations, we propose to
evaluate them based on their effectiveness in real-
istic in-context retrieval tasks. This offers a more
general and realistic measure of retrieval capability,
2We note NIAH task can also be viewed as a special case
of this formulation, where the gold document set only contains
one document (the needle).
as it captures semantic relevance rather than relying
solely on verbatim copying.
Query-focused retrieval score (QRscore).We
use QRscore as a measure of the retrieval capabil-
ity of an attention head in response to a specific
query. Formally, let h∈ H be an attention head
within the language model, and let Ah denote the
attention weights (post-softmax) of head h over
a query prompt {D, q}, such as a prompt with a
book followed by a question over its contents. The
query-focused attention scores of head h towards a
documentd i is calculated as follows:
QRscoreh(q, di) = 1
|q|
X
tq ∈q
X
td∈di
A
tq →td
h ,(2)
where tq denotes tokens in the query q, td repre-
sents tokens in the document di, and Atq→td
h is the
attention weight of h from tq to td. This formula-
tion quantifies the degree to which headhfocuses
on document di in response to q. Lastly, we aggre-
gate the scores for all documents di within the gold
document set D∗
[q], resulting in the final QRscore
for headhwith respect to the queryq:
QRscoreh(q) = 1
|q|
X
di∈D∗
X
tq ∈q
X
td∈di
A
tq →td
h (3)
3.2 Detecting QRHEADon Real-World Tasks
With the QRscore defined in Eq. 3, we can now
quantify the retrieval capabilities of each attention
head over a given set of documents in response to
a query. To achieve this, we leverage a head de-
tection dataset T={(q, D, D ∗
[q])}, which consists
of a query q, a set of candidate documents D, and
the corresponding gold documents D∗
[q]. Notably,
our approach does not require explicit answers to
the queries—only the annotations of the gold docu-
ment. Using this detection dataset T , we compute

<!-- page 4 -->

the effectiveness of an attention headh for retrieval
as follows:
QRscoreh,T = 1
|T |
X
(q,D,D ∗)∈T
QRscoreh(q)(4)
As shown in Figure 2, instead of synthetic needle-
in-a-haystack task (NIAH) (Kamradt, 2023), we
use more realistic in-context retrieval task for head
detection (e.g., claim verification over books). We
argue that more natural and realistic distractors pro-
vide more effective supervision that allows identify-
ing heads that are better at differentiating relevant
context from distracting context. We also note that
even a small amount (< 100) of realistic data points
can be sufficient, allowing us to find QRHEAD
heads that contribute to improved downstream per-
formance.
3.3 Comparing QRHEADand Original
Retrieval Head
We have introduced our method for detecting QR-
HEAD. Here, we compare the QRHEADwith
original retrieval head (RETHEAD) within the
same model, using Llama-3.1-8B-Instruct (Llama-
3 Team, 2024) as a case study.
First, following the analysis setup of Wu et al.
(2025b), we measure the impact of pruning by the
performance drop on NIAH test. Specifically, we
prune the top 32 heads (roughly 3% of all attention
heads in LLaMA-3.1-8B), following the commonly
reported 5% sparsity level of retrieval heads in Wu
et al. (2025b); Zhao et al. (2025). As shown in
Figure 1, pruning the top 32 QRHEADresults in
near-complete failure on the NIAH performance,
whereas pruning the top 32 RETHEADyields a
much smaller performance decline. 3 In addition,
we findsubstantial divergencebetween the two
sets. Among the top 32 and top 64 heads, only 8
and 32 overlap, respectively. This less than 25%
overlap in the top 32 highlights the distinct roles of
QRHEADand RETHEAD.
4 Using QRHEADto Build A
General-Purpose Retriever
In this section, we describe how the detected
QRHEADcan be used in downstream applica-
tions. Specifically, we find the attention mass of
QRHEADprovides highly reliable signals for in-
context retrieval.
3See Appendix B for results on Qwen-2.5-7B-Instruct.
4.1 The Method
Given a selected set of QRHEAD Hselect, a query
q, and a collection of passages D, we compute the
retrieval score for each passage di by aggregating
the QRscore across all heads inH select:
R(q, di) = 1
|Hselect|
X
h∈Hselect
QRscoreh(q, di).(5)
Passages are then ranked using their retrieval scores.
We call our retrieval system QRRETRIEVER. It
offers several advantages: (1)General-purpose:
applicable across diverse domains without train-
ing, unlike traditional retrievers that often are often
limited in generalizing out of domain (2)Model-
agnostic:compatible with any transformer-based
LMs without modification, (3)Efficient:leverages
attention patterns to process long context simulta-
neously without expensive generation or pairwise
comparisons.
Calibration.To mitigate intrinsic biases in LMs’
attention weights, we adopt the score calibration
method proposed by Chen et al. (2025). Instead of
directly using R(q, di) as the score, we addition-
ally compute baseline score, R(qnull, di), using a
context-free null query qnull ("N/A"). For each di,
we use calibrated the score R(q, di)−R(q null, di)
as the final retriever score.
4.2 Applications
Long-context reasoning.Long-context lan-
guage models often struggle with performance
degradation when processing long context (Yen
et al., 2025; Ye et al., 2025; Liu et al., 2024a).
To address this, we integrate QRRETRIEVER
within a retrieval-augmented generation (RAG)
framework. Given a long-context input and a query,
we segment the input into smaller chunks and use
QRRETRIEVERto score and subsequently extract
the most relevant ones. The extracted context
are concatenated to create a reduced context, that
is then given to the LM for generating the final
answer in another forward pass.
Passage re-ranking.Text retrieval powers
many retrieval-augmented downstream applica-
tions (Lewis et al., 2020). A critical component
in the retrieval pipeline is the re-ranker, which re-
orders the passages returned by a first-stage re-
triever to enhance top passage relevance (Nogueira
and Cho, 2020; Ma et al., 2024). QRRETRIEVER
can naturally be used as a re-ranker as part of any

<!-- page 5 -->

LongMemEval CLIPPER
RETRIEVER
RETRIEVALEND-TO-ENDRETRIEVALEND-TO-END
RECALL@KPERFORMANCERECALL@KPERFORMANCE
k = 5 k = 10 Top-5 Top-10 k = 3 k = 5 Top-3 Top-5
Base LM: Llama-3.2-3B-Instruct
Full context - - 28.1 - - 25.2
BM25 57.5 67.5 46.1 44.9 74.6 83.7 20.0 22.8
Contriever 62.7 79.248.646.5 60.2 78.9 12.6 18.4
Stella 63.9 77.6 44.947.783.3 90.0 21.3 25.1
RankGPT 1.8 3.4 23.5 23.3 16.8 27.3 3.6 8.8
RankGPTBubble 2.1 3.8 24.0 24.4 17.0 27.4 3.8 8.8
ICR 68.7 78.8 46.5 45.1 72.8 83.6 19.4 23.6
QRRETRIEVER(Ours)77.6 86.647.447.7 85.5 93.423.426.9
Base LM: Llama-3.1-8B-Instruct
Full context - - 46.5 - - 31.3
BM25 57.5 67.5 48.8 50.9 74.6 83.7 37.9 37.9
Contriever 62.7 79.2 52.6 55.4 60.2 78.9 28.2 31.1
Stella 63.9 77.6 50.9 58.4 83.3 90.0 38.8 39.6
RankGPT 2.1 4.0 26.7 24.2 30.0 39.4 15.9 19.4
RankGPTBubble 8.3 9.0 28.1 27.0 36.7 44.3 19.7 20.4
ICR 77.0 84.4 59.3 56.1 89.3 94.7 43.842.5
QRRETRIEVER(Ours)85.5 91.7 59.8 60.2 93.8 96.9 47.641.9
Base LM: Llama-3.1-70B-Instruct
Full context - - 34.2 - - 63.9
BM25 57.5 67.5 52.8 53.0 74.6 83.7 60.1 66.5
Contriever 62.7 79.2 53.7 60.5 60.2 78.9 38.5 49.7
Stella 63.9 77.6 56.3 62.3 83.3 90.0 65.9 71.2
RankGPT 1.8 3.5 21.2 27.4 57.0 63.4 44.7 50.4
RankGPTBubble 47.9 49.0 44.0 42.6 74.3 78.8 58.4 61.5
ICR 32.1 46.5 36.5 39.8 86.2 93.1 68.9 71.6
QRRETRIEVER(Ours)80.4 88.5 66.7 67.7 95.0 98.2 74.2 73.3
Table 1: Results on LongMemEval and CLIPPER. The base model denotes the LM used for both the retriever and
end-to-end generation. QRHEADused for CLIPPERare found through using LongMemEval.
retrieval pipeline without any fine-tuning by simply
concatenating the retrieved passages in the input
and scoring their relevance directly.
5 Experiments
We evaluate QRRETRIEVERon two tasks: long-
context reasoning (§5.2) and re-ranking (§5.3).
5.1 Base Models and Baselines
Base LMs.We experiment with open-weight,
instruction-tuned LMs from two families across
different sizes, including Llama-3.2 (3B), Llama-
3.1 (8B and 70B) of Llama family (Llama-3 Team,
2024), and Qwen2.5 (7B) of Qwen family (Yang
et al., 2024). With QRRETRIEVER, we use 16
heads for models with fewer than 10B parameters,
and 32 heads for Llama-3.1-70B. This corresponds
to approximately 1–2% of the total attention heads,
given the sparsity of retrieval heads.
Baselines.We compare our methods against sev-
eral strong baselines. Following Wu et al. (2025a),
we compare against dense retrievers, including
Contriever (Izacard et al., 2022) and 1.5B Stella
V5 (Zhang et al., 2025), two popular strong dense
retrievers. For Contriever, we truncate the input
to 512 tokens according to its maximum context
length. We also compare against existing LLM-
based re-rankers, including:
• RankGPT(Sun et al., 2024) is a generative
re-ranker that instructs LLMs to output the
ranking order of a given set of documents
based on a query. We experiment with two
variants of RankGPT: (1)RankGPT with-
out sliding window, which directly inputs all
documents into the model prompt simultane-
ously, and (2)RankGPT with sliding window
(RankGPTBubble), which leverages bubble sort to
rank smaller subsets of documents incrementally.
• In-Context-Re-ranking(ICR; Chen et al.,
2025) is a re-ranker that also leverages the at-
tention for relevance scoring. ICR uses full atten-
tion heads for scoring relevace, whereas we only
use the attention weights of selected QRHEAD.
5.2 Long-Context Multi-Hop Reasoning
Datasets.We use 1)LongMemEval(Wu et al.,
2025a), which evaluates the long-term memory ca-
pabilities of LLM-driven chat assistants, and 2)
CLIPPER(Pham et al., 2025), which evaluate
claim-verification over books. Both datasets fea-
ture long-context (90K to 120K) and require multi-
hop reasoning over several pieces of evidences. We

<!-- page 6 -->

NQ COVID NFCorpus FiQA Scifact Scidocs FEVER Climate DBPedia Robust04 News A vg
BM25 30.5 59.5 32.2 23.6 67.9 14.9 65.1 16.5 31.8 40.7 39.5 38.4
Base LM: Llama-3.2-3B-Instruct
RankGPT 30.0 59.5 32.2 23.6 67.9 14.9 65.9 17.1 31.8 40.7 39.5 38.5
RankGPTBubble 33.2 61.8 32.0 22.4 66.1 14.8 65.8 17.1 34.8 40.5 40.2 39.0
ICR 49.2 72.3 33.8 31.8 73.3 17.4 82.6 24.2 34.7 47.2 44.7 46.5
QRRETRIEVER(Ours)54.9 77.4 35.1 35.1 74.7 18.3 83.7 24.5 36 49.7 45.1 48.6
Base LM: Llama-3.1-8B-Instruct
RankGPT 30.0 59.5 32.2 23.6 67.9 14.9 65.9 16.8 31.8 40.7 39.5 38.4
RankGPTBubble 53.7 75.5 34.3 31.4 69.3 17.4 67.5 23.842.947.846.246.3
ICR 54.0 73.3 34.8 35.6 75.5 19.085.8 24.836.9 49.0 44.5 48.5
QRRETRIEVER(Ours)58.6 77.5 35.3 39.1 76.2 19.485.3 23.9 37.251.4 46.2 50.0
Base LM: Qwen-2.5-7B-Instruct
RankGPT 30.0 59.5 32.2 23.6 67.9 14.9 65.9 16.8 31.8 40.7 39.5 38.4
RankGPTBubble 42.770.5 34.1 29.569.316.670.5 19.737.1 46.4 43.643.6
ICR 43.1 66.1 32.7 27.071.116.4 79.2 19.6 35.3 43.0 40.0 43.0
QRRETRIEVER(Ours)49.967.7 33.1 29.2 71 15.380.7 20.135.7 43.7 39.844.2
Base LM: Llama-3.1-70B-Instruct
RankGPT 45.4 62.7 33.6 28.6 71.3 16.1 74.2 18.9 37.6 41.3 39.8 42.7
RankGPTBubble 58.481.2 36.141.0 76.1 20.2 80.025.1 45.5 59.0 48.5 51.9
ICR 57.9 72.3 34.2 38.9 74.6 19.486.422.0 38.3 42.6 40.4 47.9
QRRETRIEVER(Ours)62.173.9 34.743.8 76.8 20.485.9 23.1 35.7 51.6 43.5 50.1
Dual Encoder and Cross Encoder
Contriever 44.6 67.5 32.8 28.4 67.1 18.9 64.2 28.0 39.5 45.7 41.7 43.5
GTR-t5-base 51.4 74.8 32.5 34.7 62.1 15.8 72.9 26.8 37.1 46.1 42.8 45.2
BGE-Reranker-base 55.2 66.4 31.0 31.7 70.8 15.7 88.6 36.5 42.5 39.9 37.0 46.8
msmarco-MiniLM 55.8 74.3 35.2 35.1 68.5 17.5 80.4 25.5 45.3 47.9 43.0 48.0
Table 2: Performance comparison (nDCG@10) on BEIR benchmarks across LMs. QRRETRIEVERgenerally
outperforms other baselines across all models. With Llama-3.1-70B, QRRETRIEVERunderperforms RankGPT
with (Bubble sort), which requires substantial amount of LLM generation calls. We also report the performance of
popular dual encoders and cross encoders for reference (gray box).
segment each dataset according to its natural struc-
ture (e.g., message in multi-turn conversation or
chapters in a book). For evaluation, we measure
retrieval performance using recall and assess down-
stream task performance with accuracy. Please
refer to Appendix A for more details.
Data for head detection.We detect QRHEAD
using a small subset of data from LongMemEval,
specifically the single-session-user subset (which
only requires single-hop reasoning) consisting of
70 examples, which we exclude from downstream
evaluation. We use the set of heads for both Long-
MemEval and CLIPPER, testing generalization to
multi-hop reasoning.
QRRETRIEVERachieves strong retrieval per-
formance for long context, leading to improved
end-to-end performance.Table 1 demonstrates
the strong performance of QRRETRIEVERon both
LongMemEval and CLIPPER: it outperforms other
baselines regarding both retrieval recall and end-
to-end performance. For instance, Llama-3.1-8B-
Instruct as the base LM, we see end-to-end perfor-
mance improvements of over 10% on both tasks
with Llama-3.1-8B-Instruct.
QRRETRIEVERgeneralizes across domains.
QRRETRIEVERoutperform off-the-shelf dense re-
trievers (Contriever and Stella) by a large margin on
LongMemEval and CLIPPER. In particular, none
of these methods are trained or calibrated on CLIP-
PER. The better performance of QRRETRIEVER
suggests its stronger cross-domain generalization
capabilities than dense retrievers.
It is worth noting that all test questions in Long-
MemEval are multi-hop, yet QRRETRIEVERper-
forms well on them despite only using single-hop
questions to detect QRHEAD.
QRRETRIEVERscales with model sizes.We
note that LM-based re-rankers show inconsis-
tent performance patterns across model scales:
RankGPT achieves near-zero retrieval recall with
small models, and retrieval performance of ICR
sees significant degradation when scaling up model
size from 8B to 70B. At the same time, the perfor-
mance of QRRETRIEVERgenerally improves as
the model size scales up.

<!-- page 7 -->

Compact LMs exhibit strong retrieval capabil-
ities despite their limited generation abilities.
As shown in Table 1, on LongMemEval, Llama-3.2-
3B-Instruct achieves a Recall@10 of 86.6, closely
matching the 88.5 score of the much larger LlamA-
3.1-70B. However, Llama-3.2-3B only achieves
a final end-to-end performance of 47.7, largely
lagging 70B’s performance of 67.7. We hypoth-
esize that the long-context limitations of compact
models stem more from their generation capabili-
ties than from their retrieval abilities. These find-
ings open up promising future directions. Compact
LMs could serve as efficient long-context retrievers,
paired with larger models for the actual generation.
5.3 Passage Re-Ranking
To test the general applicability of QRRE-
TRIEVER, we evaluate our method on BEIR bench-
mark (Thakur et al., 2021) consisting of diverse do-
mains. We compare against zero-shot LLM-based
re-rankers, RankGPT and ICR. We also report the
performance of popular dual encoders and cross
encoders from sentence transformer for references
(please refer to Appendix C for details about these
models).
Setting.Our setting largely follows prior
work (Chen et al., 2025). We re-rank 200 passages
retrieved using BM25, resulting a overall context
length ranging from 16K to 64K depending on the
average document length of domains. We report the
performance on the set of tasks used in Chen et al.
(2025), we sub-sampled 512 random questions for
each domain for evaluation.
Data for head detection.For BEIR, we utilize
the 256 (held-out) data points from NQ and use
them for on all other domains zero-shot.
Results.Table 2 summarizes the BEIR results,
demonstrating the strong effectiveness of QRRE-
TRIEVERas a general-purpose retriever. For mod-
els under 10B parameters, QRRETRIEVERconsis-
tently outperforms other baselines. With Llama-
3.1-8B, it achieves an average score of 50.0, out-
performing RankGPT by 3.7 points and ICR by 1.5
points. For the larger Llama-3.1-70B model, QR-
RETRIEVERsignificantly surpasses ICR, though
it generally lags RankGPT Bubble (which require
over 200 generation calls). Nevertheless, QRRE-
TRIEVERachieves the best performance on several
domains, such as SciFact and FiQA. In addition,
we find that QRRETRIEVERdirectly benefits from
BEIR SHUFFLED LONGMEMEVAL
NDCG@10 RECALL
Llama-8B
RANDOMHEADS37.5 59.8
FULLHEADS42.8 77.0
RETRIEVALHEAD43.4 81.5
QRHEAD47.5 85.5
Qwen-7B
RANDOMHEADS19.9 57.2
FULLHEADS22.6 67.1
RETRIEVALHEAD27.4 70.7
QRHEAD31.9 83.2
Table 3: Comparison across head selection strategies.
Using QRHEADsubstantially outperforms using all
heads or using original retrieval heads.
the general-purpose capabilities of strong LLMs,
outperforming popular dual encoders and cross-
encoders that fine-tune various base models to im-
prove retrieval performance.
6 Analysis
In this section, we analyze the key advantages of
QRHEAD(e.g., length generalization) and exam-
ine the factors underlying its effectiveness. Addi-
tional analyses on the impact of varying the number
of heads (Appendix F) and inference latency (Ap-
pendix E) are provided in the appendix.
6.1 Impact of Head Selection
We provide further ablations on head selection, the
core idea behind QRRETRIEVER. We experiment
with different sets of heads, including (1) using
our QRHEAD, (2) using all the attention heads,
(3) using original retrieval head, and (4) using ran-
domly selected heads. We use 16 heads for all set-
tings. Table 3 presents the retrieval performance on
LongMemEval re-ranking performance on BEIR
(aggregated across tasks).4 The performance gaps
between different strategies demonstrate the impor-
tance of using the right heads for retrieval. Using
original retrieval heads is effective, compared to
using random heads or full heads. Using our im-
proved QRHEADconsistently outperforms using
original retrieval heads.
6.2 Generalizability Across Lengths
We test the length generalization of QRHEAD: if
we detect QRHEADon relatively short context
length (32K), can the heads generalize to longer
context lengths (128K)?
We test such short-to-long generalization by
controlling the number of documents in context.
4Here, we use BEIR where input documents are randomly
shuffled rather than ranked by BM25. This setup allows uni-
form evaluation of retrieval across the full context.

<!-- page 8 -->

Model: LLama-3.1-8B-Instruct
NQ+FEVER LongMemEval
32K 128K 32K 128K
ICR 66.7 56.5 85.2 78.2
QRRETRIEVER 32K 70.163.9 89.2 85.2
QRRETRIEVER 128K 68.867.2 89.2 85.6
Model: Qwen-2.5-7B-Instruct
NQ+FEVER LongMemEval
32K 64K 32K 64K
ICR 40.0 17.4 83.4 67.1
QRRETRIEVER 32K 51.9 25.390.2 77.9
QRRETRIEVER 64K 54.1 29.190.1 77.0
Table 4: Results on short-to-long generalization of QR-
HEAD. QRHEADdetected with relative short-context
data can be used for retrieval on longer context.
Data BEIR LONGMEM
NDCG@10 RECALL
Model: LLama-3.1-8B-Inst
QRHEADNQ47.583.9
QRHEADLongMem 47.185.6
QRHEADNIAH 46.8 83.4
RETRIEVALHEADNIAH 43.4 81.5
Model: Qwen-2.5-7B-Inst
QRHEADNQ 31.9 80.2
QRHEADLongMem32.1 83.2
QRHEADNIAH 30.9 79.7
RETRIEVALHEADNIAH 27.4 70.7
Table 5: Analysis of factors contributing to improved
head selection. Applying QRScore (§3.1) on NIAH
results in more effective heads than the original retrieval
heads. Using QRScore on realistic tasks yields the most
effective head selection overall.
This results in datasets of different lengths ranging
from 32K to 128K tokens. We detect QRHEAD
from both short and long datasets and test their
performance on re-ranking tasks (using two rep-
resentative subsets: NQ and FEVER) and Long-
MemEval. For Qwen-2.5-7B, we set the longer
context length to 64K due to its original 32K limit.
As shown in Table 4, QRHEADdetected using
short-context data can generalize to longer-context
settings, though heads detected from longer data
generally yield better long-context performance.
6.3 Ablations on Scoring Function and Task
Selection for Head Detection
In §3, we describe two key factors for head detec-
tion: using query-context attention objective, and
using realistic data. To assess the importance of
these factors, we experiment with detecting heads
on NIAH using QRscore (§3). As shown in Table 5,
applying QRscore on NIAH leads to improved per-
formance compared to using the original retrieval
heads detected from the same task. However, us-
ing realistic tasks such as NQ and LongMemEval
with QRscore yields the best overall performance.
These results highlight the importance of both the
Overlap (Top 64) BEIR
Set0 Set1 Set2 nDCG@10
Model: LLama-3.1-8B-Inst
QRHEAD Set0 64 51 51 49.8
QRHEAD Set1 51 64 53 49.7
QRHEAD Set2 51 53 64 49.9
Model: Qwen-2.5-7B-Inst
QRHEAD Set0 64 50 53 44.2
QRHEAD Set1 50 64 57 44.4
QRHEAD Set2 53 57 64 44.5
Table 6:Left: Overlap in QRHEADidentified using
three disjoint sets of 128 random samples from NQ.
Right: BEIR performance (nDCG@10) using QR-
HEADdetected from each sample set.
scoring method and head detection data.
6.4 Sensitivity of QRHEADDetection to
Variation in Detection Data
In Section 5.3, we show using a small number of
samples from NQ is sufficient to identify effec-
tive QRHEADfor BEIR re-ranking tasks. We as-
sess the robustness of this head detection process
to different random samples of detection set, by
experimenting with three disjoint random subsets
of NQ, each containing 128 examples. Table 6
presents the overlap among the top-64 heads se-
lected from these subsets and their performance on
BEIR benchmark. Across two LLMs from differ-
ent model families (Llama and Qwen), we observe
a high degree of consistency with over 50 heads
overlapping among the top 64 across subsets. Fur-
thermore, the downstream performance remains
stable across these variations. These results indi-
cate that QRRETRIEVERcan be reliably identified
using a small sample of data.
7 Related Work
LM-based retrieval and re-ranking.LMs are
widely used in retrieval, including embedding-
based methods (Muennighoff, 2022; Lee et al.,
2021) and generative approaches (Tay et al., 2022;
Cao et al., 2021; Sun et al., 2023). For re-ranking,
instruction-tuned LMs been adapted as re-rankers
in various ways (Sun et al., 2024; Drozdov et al.,
2023; Sachan et al., 2023; Ma et al., 2023; Pradeep
et al., 2023), leveraging their generation capabil-
ities. Similar to our approach, recent work has
explored using logits (Reddy et al., 2024) or ag-
gregated attention scores (Chen et al., 2025) for
re-ranking. In contrast, we identify a specialized
set of attention heads responsible for retrieval, of-
fering improved performance and interpretability.

<!-- page 9 -->

Localizing model behavior .Interpretability stud-
ies have shown that many core behaviors of LMs,
including in-context learning (Olsson et al., 2022;
Todd et al., 2024; McDougall et al., 2023) and
retrieval (Wu et al., 2025b), can be traced to spe-
cialized transformer modules (Meng et al., 2022;
Dai et al., 2022; Stolfo et al., 2024). Techniques
have been proposed to localize such modules with
a small amount of data (Meng et al., 2022; Geiger
et al., 2024; Bhaskar et al., 2024), and to intervene
on them for control (Li et al., 2023; Yin et al., 2024;
Huang et al., 2025) or efficiency (Tang et al., 2025;
Xiao et al., 2025; Liu et al., 2024b). However, only
a few works (Zhao et al., 2025) have examined at-
tention head specialization in long-context settings,
where attention is known to be not robust (Liu et al.,
2024a; Xiao et al., 2024a), and it is an open ques-
tion if intervening the localized modules is cru-
cial in practical settings (Hase et al., 2023; Wang
and Veitch, 2024). Our work contributes to this
line of research by finding better specialized set
of attention heads that explain the model behav-
ior for query-focused long-context retrieval, and
that can be practically useful for zero-shot efficient
retrieval.
8 Conclusion
We introduced Query-Focused Retrieval Heads
(QRHEAD), a set of attention heads specialized
in identifying query-relevant information in long-
context inputs. Detected using query-context atten-
tion scores on realistic data, QRHEADare better
aligned with practical retrieval tasks than original
retrieval heads. Built on top of QRHEAD, our
retrieval method QRRETRIEVERachieves strong
performance on both long-context reasoning and
re-ranking tasks, outperforming dense retrievers
and other LLM-based re-rankers in many settings.
These findings highlight the practical utility of QR-
HEADand offer insights for further improving re-
trieval with LMs.
Limitations
Our work detects improved retrieval heads and
builds general-purpose retrievers based on them.
We do not explore techniques that involve updat-
ing model parameters, as our goal is to develop
flexible methods that can directly use off-the-shelf
models as retrievers. Consequently, we leave to fu-
ture work the investigation of parameter-updating
techniques that leverage insights from QRHEAD.
While our method finds that QRHEADcan en-
hance downstream performance, and shows the
importance of two factors leading to selection of
better heads. We lack a complete understanding of
the internal mechanism accounting for QRHEAD’s
effectiveness. Future work could apply circuit anal-
ysis techniques (e.g., Bhaskar et al. (2024); Shi
et al. (2024)) to dissect the fine-grained behaviors
and roles of these heads.
Our evaluation primarily targets passage re-
ranking and long-context multi-hop reasoning tasks.
Although our approach is conceptually applica-
ble to broader long-context tasks—such as long-
document summarization (Shaham et al., 2023; La-
ban et al., 2024)—it remains unclear whether it
generalizes to such tasks without thorough empiri-
cal validation.
Finally, our experiments are limited to English
datasets. As LMs may exhibit different behaviors
across languages, the cross-lingual robustness of
our approach remains an open question.
9 Acknowledgments
This work is gratefully supported by an NSF CA-
REER award (IIS-2239290) and a grant from Intel.
Howard Yen is supported by the William A. Dippel’
50 * 55 Graduate Fellowship.
References
Adithya Bhaskar, Alexander Wettig, Dan Friedman, and
Danqi Chen. 2024. Finding transformer circuits with
edge pruning. InThe Thirty-eighth Annual Confer-
ence on Neural Information Processing Systems.
Nicola De Cao, Gautier Izacard, Sebastian Riedel, and
Fabio Petroni. 2021. Autoregressive entity retrieval.
Preprint, arXiv:2010.00904.
Shijie Chen, Bernal Jiménez Gutiérrez, and Yu Su. 2025.
Attention in large language models yields efficient
zero-shot re-rankers.Preprint, arXiv:2410.02642.
Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao
Chang, and Furu Wei. 2022. Knowledge neurons in
pretrained transformers. InProceedings of the 60th
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 8493–
8502, Dublin, Ireland. Association for Computational
Linguistics.
Andrew Drozdov, Honglei Zhuang, Zhuyun Dai, Zhen
Qin, Razieh Rahimi, Xuanhui Wang, Dana Alon,
Mohit Iyyer, Andrew McCallum, Donald Metzler,
and Kai Hui. 2023. Parade: Passage ranking us-
ing demonstrations with large language models.
Preprint, arXiv:2310.14408.

<!-- page 10 -->

Atticus Geiger, Zhengxuan Wu, Christopher Potts,
Thomas Icard, and Noah Goodman. 2024. Finding
alignments between interpretable causal variables
and distributed neural representations. InProceed-
ings of the Third Conference on Causal Learning and
Reasoning, volume 236 ofProceedings of Machine
Learning Research, pages 160–187. PMLR.
Peter Hase, Mohit Bansal, Been Kim, and Asma Ghan-
deharioun. 2023. Does localization inform editing?
surprising differences in causality-based localization
vs. knowledge editing in language models. InThirty-
seventh Conference on Neural Information Process-
ing Systems.
Lei Huang, Xiaocheng Feng, Weitao Ma, Yuchun Fan,
Xiachong Feng, Yangfan Ye, Weihong Zhong, Yux-
uan Gu, Baoxin Wang, Dayong Wu, Guoping Hu, and
Bing Qin. 2025. Improving contextual faithfulness
of large language models via retrieval heads-induced
optimization.arXiv preprint arXiv:2501.13573.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2022. Unsupervised dense infor-
mation retrieval with contrastive learning.Preprint,
arXiv:2112.09118.
Garrett Kamradt. 2024. Needle in a haystack - pressure
testing llms.
Gregory Kamradt. 2023. Needle In A Haystack - pres-
sure testing LLMs.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Philippe Laban, Alexander Fabbri, Caiming Xiong, and
Chien-Sheng Wu. 2024. Summary of a haystack:
A challenge to long-context LLMs and RAG sys-
tems. InProceedings of the Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 9885–9903. Association for Computational
Linguistics.
Jinhyuk Lee, Alexander Wettig, and Danqi Chen.
2021. Phrase retrieval learns passage retrieval, too.
Preprint, arXiv:2109.08133.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter
Pfister, and Martin Wattenberg. 2023. Inference-
time intervention: Eliciting truthful answers from
a language model. InThirty-seventh Conference on
Neural Information Processing Systems.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024a. Lost in the middle: How language
models use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Wenhan Liu, Xinyu Ma, Yutao Zhu, Ziliang Zhao,
Shuaiqiang Wang, Dawei Yin, and Zhicheng Dou.
2024b. Sliding windows are not the end: Exploring
full ranking with long-context large language models.
Preprint, arXiv:2412.14574.
Llama-3 Team. 2024. The Llama 3 herd of models.
arXiv preprint arXiv:2407.21783.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2024. Fine-tuning llama for multi-stage
text retrieval. InProceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, SIGIR ’24, page
2421–2425, New York, NY , USA. Association for
Computing Machinery.
Xueguang Ma, Xinyu Zhang, Ronak Pradeep, and
Jimmy Lin. 2023. Zero-shot listwise document
reranking with a large language model.Preprint,
arXiv:2305.02156.
Callum McDougall, Arthur Conmy, Cody Rushing,
Thomas McGrath, and Neel Nanda. 2023. Copy
Suppression: Comprehensively Understanding an At-
tention Head.arXiv preprint arXiv:2310.04625.
Kevin Meng, David Bau, Alex J Andonian, and Yonatan
Belinkov. 2022. Locating and editing factual associ-
ations in GPT. InAdvances in Neural Information
Processing Systems.
Niklas Muennighoff. 2022. Sgpt: Gpt sen-
tence embeddings for semantic search.Preprint,
arXiv:2202.08904.
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo
Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan,
Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022.
Large dual encoders are generalizable retrievers. In
Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing.
Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage
re-ranking with bert.Preprint, arXiv:1901.04085.

<!-- page 11 -->

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas
Joseph, Nova DasSarma, Tom Henighan, Ben Mann,
Amanda Askell, Yuntao Bai, Anna Chen, Tom Con-
erly, Dawn Drain, Deep Ganguli, Zac Hatfield-
Dodds, Danny Hernandez, Scott Johnston, Andy
Jones, Jackson Kernion, Liane Lovitt, and 7 oth-
ers. 2022. In-context Learning and Induction Heads.
arXiv preprint arXiv:2209.11895.
Chau Minh Pham, Yapei Chang, and Mohit Iyyer. 2025.
Clipper: Compression enables long-context synthetic
data generation.Preprint, arXiv:2502.14854.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy
Lin. 2023. Rankzephyr: Effective and robust zero-
shot listwise reranking is a breeze!Preprint,
arXiv:2312.02724.
Revanth Gangi Reddy, JaeHyeok Doo, Yifei Xu,
Md Arafat Sultan, Deevya Swain, Avirup Sil, and
Heng Ji. 2024. First: Faster improved listwise
reranking with single token decoding.Preprint,
arXiv:2406.15657.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing. Associa-
tion for Computational Linguistics.
Devendra Singh Sachan, Mike Lewis, Mandar Joshi,
Armen Aghajanyan, Wen tau Yih, Joelle Pineau,
and Luke Zettlemoyer. 2023. Improving passage re-
trieval with zero-shot question generation.Preprint,
arXiv:2204.07496.
Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant,
and Omer Levy. 2023. ZeroSCROLLS: A zero-shot
benchmark for long text understanding. InFindings
of the Conference on Empirical Methods in Natural
Language Processing (EMNLP Findings).
Claudia Shi, Nicolas Beltran-Velez, Achille Nazaret,
Carolina Zheng, Adrià Garriga-Alonso, Andrew Jes-
son, Maggie Makar, and David Blei. 2024. Hypoth-
esis testing the circuit hypothesis in LLMs. InThe
Thirty-eighth Annual Conference on Neural Informa-
tion Processing Systems.
Alessandro Stolfo, Ben Peng Wu, Wes Gurnee, Yonatan
Belinkov, Xingyi Song, Mrinmaya Sachan, and Neel
Nanda. 2024. Confidence regulation neurons in lan-
guage models. InThe Thirty-eighth Annual Confer-
ence on Neural Information Processing Systems.
Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang
Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen,
Dawei Yin, Maarten de Rijke, and Zhaochun Ren.
2023. Learning to tokenize for generative retrieval.
Preprint, arXiv:2304.04171.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2024. Is chatgpt good at search?
investigating large language models as re-ranking
agents.Preprint, arXiv:2304.09542.
Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Danning
Ke, Shikuan Hong, Yiwu Yao, and Gongyi Wang.
2025. Razorattention: Efficient KV cache compres-
sion through retrieval heads. InThe Thirteenth Inter-
national Conference on Learning Representations.
Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo
Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui,
Zhe Zhao, Jai Gupta, Tal Schuster, William W.
Cohen, and Donald Metzler. 2022. Transformer
memory as a differentiable search index.Preprint,
arXiv:2202.06991.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. InThirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 2).
Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron
Mueller, Byron C Wallace, and David Bau. 2024.
Function vectors in large language models. InThe
Twelfth International Conference on Learning Repre-
sentations.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. InAdvances in Neural Information Pro-
cessing Systems, volume 30. Curran Associates, Inc.
Zihao Wang and Victor Veitch. 2024. Does editing
provide evidence for localization? InICML 2024
Workshop on Mechanistic Interpretability.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-
Wei Chang, and Dong Yu. 2025a. Longmemeval:
Benchmarking chat assistants on long-term interac-
tive memory.Preprint, arXiv:2410.10813.
Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao
Peng, and Yao Fu. 2025b. Retrieval head mechanis-
tically explains long-context factuality. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, junxian
guo, Shang Yang, Haotian Tang, Yao Fu, and Song
Han. 2025. Duoattention: Efficient long-context
LLM inference with retrieval and streaming heads. In
The Thirteenth International Conference on Learning
Representations.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. 2024a. Efficient streaming
language models with attention sinks. InThe Twelfth
International Conference on Learning Representa-
tions.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024b. C-
pack: Packed resources for general chinese embed-
dings. InProceedings of the 47th international ACM
SIGIR conference on research and development in
information retrieval, pages 641–649.

<!-- page 12 -->

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and
39 others. 2024. Qwen2 technical report.ArXiv,
abs/2407.10671.
Sohee Yang and Minjoon Seo. 2020. Is retriever merely
an approximator of reader?ArXiv, abs/2010.10999.
Xi Ye, Fangcong Yin, Yinghui He, Joie Zhang, Howard
Yen, Tianyu Gao, Greg Durrett, and Danqi Chen.
2025. Longproc: Benchmarking long-context lan-
guage models on long procedural generation. In
arXiv.
Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding,
Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and
Danqi Chen. 2025. Helmet: How to evaluate long-
context language models effectively and thoroughly.
Fangcong Yin, Xi Ye, and Greg Durrett. 2024. Lofit:
Localized fine-tuning on LLM representations. In
The Thirty-eighth Annual Conference on Neural In-
formation Processing Systems.
Dun Zhang, Jiacheng Li, Ziyang Zeng, and Fulong
Wang. 2025. Jasper and stella: distillation of sota
embedding models.Preprint, arXiv:2412.19048.
Xinyu Zhao, Fangcong Yin, and Greg Durrett. 2025.
Understanding synthetic context extension via re-
trieval heads. InProceedings of ICML.
A Details about Evaluation Datasets
We use LongMemEval (Wu et al., 2025a) and CLIP-
PER (Pham et al., 2025) for evaluating our systems
on long-context reasoning.
LongMemEvalevaluates the long-term mem-
ory capabilities of LLM-driven chat assistants
across five fundamental abilities: information ex-
traction, multi-session reasoning, temporal rea-
soning, knowledge updates, and abstention. We
segment the LongMemEval-S dataset (∼115k to-
kens/question) at the round level, where each round
is a document consisting of a single user message
paired with the corresponding assistant response.
CLIPPERtargets narrative claim verification—a
challenging long-context reasoning task that re-
quires verifying claims over entire books, with an
average length of 90K tokens and 23 chapters. In
CLIPPER, data is split at the chapter level, with
each chapter treated as an individual document dur-
ing retrieval.
Evaluation ProcessFor each question, we first
feed the entire context (e.g., all chapters or dialogue
rounds) into the language model without using any
first-stage retriever. We compute a retrieval score
for each document or segment using our method de-
scribed in §4. We then select the top-kdocuments
based the scores, concatenate them, and feed them
together with the query into the language model
in a second pass to generate the final answer. We
choose k= 5,10 for LongMemEval and k= 3,5
for Clipper. We report retrieval performance us-
ing recall and downstream task performance using
accuracy.
B NIAH Test on Qwen-2.5-7B-Instruct
We evaluate Qwen-2.5-7B-Instruct on the NIAH
test by masking selected attention heads. As shown
in Figure 3 and Figure 4, pruning the top 16 QR-
HEADleads to a more substantial degradation in
NIAH performance compared to pruning the top 16
RETHEAD, indicating the greater functional impor-
tance of QRHEAD. When pruning the top 32 heads,
the performance gap between QRHEADand RET-
HEADnarrows, suggesting that QRHEADachieves
better efficiency and effectiveness with fewer heads
for retrieval in NIAH task.

<!-- page 13 -->

1K 50K100K
0.1
0.5
0.9
Original Retrieval Heads (Wu et al., 2025)
QRHeads(Ours)0.1
0.5
0.9
Context Length
Depth
Random Heads0.1
0.5
0.9
Figure 3:Top: Masking 16 random heads of Qwen2.5-
7B-Instruct.Middle: Masking the top 16 original re-
trieval heads (Wu et al., 2025b).Bottom: Masking the
top 16 QRHeads.
C Details about Dual Encoder and Cross
Encoder Baselines on BEIR
In §5.3, we report the performance of sev-
eral traditional retrievers, including dual en-
coders (Karpukhin et al., 2020) and cross-
encoders (Yang and Seo, 2020). We use pop-
ular models from the SentenceTransformers li-
brary (Reimers and Gurevych, 2019).5 Specifically,
we include the following dual encoders:
• Contriever(Izacard et al., 2022): We use
the checkpoint from https://huggingface.
co/facebook/contriever-msmarco, which
is fine-tuned on MS MARCO.
• GTR-T5-Base(Ni et al., 2022): We use the
checkpoint from https://huggingface.co/
sentence-transformers/gtr-t5-base.
We also include two cross-encoders fine-tuned
on MS MARCO:
5https://sbert.net/
1K 50K100K
0.1
0.5
0.9QRHeads(Ours)0.1
0.5
0.9
Context Length
Depth
Random Heads0.1
0.5
0.9
Original Retrieval Heads (Wu et al., 2025)
Figure 4:Top: Masking 32 random heads of Qwen2.5-
7B-Instruct.Middle: Masking the top 32 original re-
trieval heads (Wu et al., 2025b).Bottom: Masking the
top 32 QRHeads.
• BGE-Reranker-Base(Xiao et al.,
2024b): We use the checkpoint from
https://huggingface.co/BAAI/
bge-reranker-base.
• MSMARCO-MiniLM(Reimers and
Gurevych, 2019): We use the check-
point from https://huggingface.co/
cross-encoder/ms-marco-MiniLM-L6-v2.
D Prompt Templates
We provide prompt templates used in our experi-
ments for ICR and QRRETRIEVERin Figure 5 and
rankGPT in Figure 6.
E Inference Time Comparison between
QRRETRIEVER, ICR, and RankGPT
We compare the latency of QRRETRIEVERwith
other LLM-based re-rankers (ICR and RankGPT).
In Table 7, we report both latency and perfor-
mance of Llama-3.1-8B and Llama-3.1-70B on NQ
dataset. Compared to RankGPT, QRRETRIEVERis

<!-- page 14 -->

{prompt_prefix} Here are some paragraphs:
[1] {Title 1 (if available)}
{Paragraph text 1}
[2] {Title 2 (if available)}
{Paragraph text 2}
...
Please find information that is relevant
to the following query in the paragraphs
above.
Query: {query}{prompt_suffix}
Figure 5: Prompt used for ICR and QRRETRIEVER.
{prompt_prefix} This is an intelligent
assistant that can rank passages based on
their relevancy to the query.
The following are {N} passages, each
indicated by a numbered identifier [i]. I
can rank them based on their relevance to
the query: "{query}"
[1] {Title 1 (if available)}
{Paragraph text 1}
[2] {Title 2 (if available)}
{Paragraph text 2}
...
The search query is: "{query}". I will
rank the {N} passages above based on their
relevance to the search query. The passages
will be listed in descending order using
identifiers, the most relevant passages
should be listed first and the output format
should be [] > [] > etc, e.g., [1] > [2] >
etc. Be sure to list all {N} ranked passages
and do not explain your ranking until after
the list is done. {prompt_suffix} Ranked
Passages: [
Figure 6: Prompt used for rankGPT.
significantly more time-efficient. This is primarily
because (1) QRRETRIEVERavoids autoregressive
generation, and (2) it does not rely on bubble sort,
which requires multiple rounds of generation to
compare elements sequentially within each bubble.
F Determine the Number of Heads Used
We select the number of heads based on the sparsity
level reported in prior work on retrieval heads (Wu
et al., 2025b). In Table 8, we include additional re-
sults on BEIR NQ using varying numbers of heads.
Avg Latency (s/sample) NQ
NDCG@10
Model: LLama-3.1-8B-Inst
RankGPTBubble 14.5 53.7
ICR 2.2 54.0
QRRETRIEVER2.2 58.6
Model: Llama-3.1-70B-Inst
RankGPTBubble 63.9 58.4
ICR 13.7 57.9
QRRETRIEVER13.7 62.1
Table 7: Inference time and retrieval performance com-
parison on NQ.
Model / #HeadsNQ (NDCG@10)
Llama-3.1-8B(#heads = 1024)
16 (∼1.5%) 58.6
64 (∼6%) 57.4
256 (∼20%) 56.6
Llama-3.1-70B(#heads = 5120)
32 (∼0.5%) 62.1
128 (∼2%) 61.9
512 (∼10%) 61.0
Table 8: Performance with different numbers of selected
heads.
The results show that performance remains con-
sistently strong when using a small, top-ranked
subset of heads, but degrades as more heads are in-
cluded. This trend aligns with the sparsity property
observed in retrieval heads.
G License of Datasets
The licenses datasets used in our work include:
• LongMemEval (Wu et al., 2025a) under MIT
License.
• Clipper (Pham et al., 2025) under Apache li-
cense 2.0.
• NQ (Kwiatkowski et al., 2019) under Creative
Commons Attribution Share Alike 3.0.
• BEIR (Thakur et al., 2021) under Creative
Commons Attribution Share Alike 4.0 Read
on choosealicense.com
H Computational Resources and Model
Sizes
We use Llama-3.2 (3B), Llama-3.1 (8B and
70B) (Llama-3 Team, 2024), and Qwen2.5
(7B) (Yang et al., 2024). 8B models were run using
a single NVIDIA A100 GPU with 80GB of mem-
ory, and 70B models were run using 4 A100 GPUs.
All experiments were conducted on A100-based
infrastructure.

<!-- page 15 -->

I Potential Risks of Our Work
N/A. Our work investigates the capabilities of
existing language models, without proposing
new model architectures or training procedures.
While large language models pose well-known
risks—including potential misuse, generation of
harmful content, and encoding of societal bi-
ases—our study does not introduce new risks be-
yond those already covered in the broader literature.
As such, we do not believe any specific risk miti-
gation measures are necessary for the scope of this
work.
