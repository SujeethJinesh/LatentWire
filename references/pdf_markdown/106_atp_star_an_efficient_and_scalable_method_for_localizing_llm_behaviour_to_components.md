# references/106_atp_star_an_efficient_and_scalable_method_for_localizing_llm_behaviour_to_components.pdf

<!-- page 1 -->

2024-02-23
AtP∗: An efficient and scalable method for
localizing LLM behaviour to components
János Kramár1, Tom Lieberum1, Rohin Shah1 and Neel Nanda1
1Google DeepMind
Activation Patching is a method of directly computing causal attributions of behavior to model compo-
nents. However, applying it exhaustively requires a sweep with cost scaling linearly in the number of
model components, which can be prohibitively expensive for SoTA Large Language Models (LLMs). We
investigate Attribution Patching (AtP) (Nanda, 2022), a fast gradient-based approximation to Activation
Patching and find two classes of failure modes of AtP which lead to significant false negatives.
We propose a variant of AtP called AtP∗, with two changes to address these failure modes while retaining
scalability. We present the first systematic study of AtP and alternative methods for faster activation
patching and show that AtP significantly outperforms all other investigated methods, with AtP∗ providing
further significant improvement. Finally, we provide a method to bound the probability of remaining
false negatives of AtP∗ estimates.
1. Introduction
As LLMs become ubiquitous and integrated into numerous digital applications, it’s an increasingly
pressing research problem to understand the internal mechanisms that underlie their behaviour –
this is the problem of mechanistic interpretability. A fundamental subproblem is to causally attribute
particular behaviours to individual parts of the transformer forward pass, corresponding to specific
components (such as attention heads, neurons, layer contributions, or residual streams), often at
specific positions in the input token sequence. This is important because in numerous case studies of
complex behaviours, they are found to be driven by sparse subgraphs within the model (Meng et al.,
2023; Olsson et al., 2022; Wang et al., 2022).
A classic form of causal attribution uses zero-ablation, or knock-out, where a component is deleted
and we see if this negatively affects a model’s output – a negative effect implies the component was
causally important. More recent work has generalised this to replacing a component’s activations with
samples from some baseline distribution (with zero-ablation being a special case where activations
are resampled to be zero). We focus on the popular and widely used method of Activation Patching
(also known as causal mediation analysis) (Chan et al., 2022; Geiger et al., 2022; Meng et al., 2023)
where the baseline distribution is a component’s activations on some corrupted input, such as an
alternate string with a different answer (Pearl, 2001; Robins and Greenland, 1992).
Given a causal attribution method, it is common to sweep across all model components, directly
evaluating the effect of intervening on each of them via resampling (Meng et al., 2023). However,
when working with SoTA models it can be expensive to attribute behaviour especially to small
components (e.g. heads or neurons) – each intervention requires a separate forward pass, and so the
number of forward passes can easily climb into the millions or billions. For example, on a prompt of
length 1024, there are2.7 ·109 neuron nodes in Chinchilla 70B (Hoffmann et al., 2022).
We propose to accelerate this process by using Attribution Patching (AtP) (Nanda, 2022), a faster,
approximate, causal attribution method, as a prefiltering step: after running AtP, we iterate through
the nodes in decreasing order of absolute value of the AtP estimate, then use Activation Patching to
more reliably evaluate these nodes and filter out false positives – we call thisverification. We typically
Corresponding author: janosk@google.com
© 2024 Google DeepMind. All rights reserved
arXiv:2403.00745v1  [cs.LG]  1 Mar 2024

<!-- page 2 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
care about a small set of top contributing nodes, so verification is far cheaper than iterating over all
nodes.
Our contributions:
• We investigate the performance of AtP, finding two classes of failure modes which produce false
negatives. We propose a variant of AtP called AtP∗, with two changes to address these failure
modes while retaining scalability:
–When patching queries and keys, recomputing the attention softmax and using a gradient
based approximation from then on, as gradients are a poor approximation to saturated
attention.
–Using dropout on the backwards pass to fix brittle false negatives, where significant positive
and negative effects cancel out.
• We introduce several alternative methods to approximate Activation Patching as baselines to
AtP which outperform brute force Activation Patching.
• We present the first systematic study of AtP and these alternatives and show that AtP significantly
outperformsallotherinvestigatedmethods, withAtP ∗ providingfurthersignificantimprovement.
• To estimate the residual error of AtP∗ and statistically bound the sizes of any remaining false
negatives we provide a diagnostic method, based on using AtP to filter out high impact nodes,
and then patching random subsets of the remainder. Good diagnostics mean that practitioners
may still gauge whether AtP is reliable in relevant domains without the costs of exhaustive
verification.
Finally, we provide some guidance in Section 5.4 on how to successfully perform causal attribution
in practice and what attribution methods are likely to be useful and under what circumstances.
2. Background
2.1. Problem Statement
Our goal is to identify the contributions to model behavior by individual model components. We
first formalize model components, then formalize model behaviour, and finally state the contribution
problem in causal language. While we state the formalism in terms of a decoder-only transformer
language model (Radford et al., 2018; Vaswani et al., 2017), and conduct all our experiments on
models of that class, the formalism is also straightforwardly applicable to other model classes.
Model components. We are given a modelM : 𝑋 → ℝ𝑉 that maps a prompt (token sequence)
𝑥 ∈ 𝑋 := {1, . . . , 𝑉 }𝑇 to output logits over a set of𝑉 tokens, aiming to predict the next token in the
sequence. We will view the modelM as a computational graph(𝑁, 𝐸 ) where the node set𝑁 is the
set of model components, and a directed edge𝑒 = (𝑛1, 𝑛2) ∈ 𝐸 is present iff the output of𝑛1 is a
direct input into the computation of𝑛2. We will use𝑛(𝑥) to represent theactivation(intermediate
computation result) of𝑛 when computingM (𝑥).
The choice of𝑁 determines how fine-grained the attribution will be. For example, for transformer
models, we could have a relatively coarse-grained attribution where each layer is considered a single
node. In this paper we will primarily consider more fine-grained attributions that are more expensive
to compute (see Section 4 for details); we revisit this issue in Section 5.
2

<!-- page 3 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
(a) MLP neurons, onCITY-PP.
 (b) Attention nodes, onIOI-PP.
Figure 1 | Costs of finding the most causally-important nodes in Pythia-12B using different methods,
on sample prompt pairs (see Table 1). The shading indicates geometric standard deviation. Cost
is measured in forward passes, thus each point’s y-coordinate gives the number of forward passes
required to find the top𝑥 nodes. Note that each node must be verified, thus𝑦 ≥ 𝑥, so all lines are
above the diagonal, and an oracle for the verification order would produce the diagonal line. For a
detailed description see Section 4.3.
(a) MLP neurons, onCITY-PP.
 (b) Attention nodes, onIOI-PP.
Figure 2 | Relative costs of methods across models, on sample prompt pairs. The costs are relative
to having an oracle, which would verify nodes in decreasing order of true contribution size. Costs
are aggregated using an inverse-rank-weighted geometric mean. This means they correspond to the
area above the diagonal for each curve in Figure 1 and are relative to the area under the dotted
(oracle) line. See Section 4.2 for more details on this metric. Note that GradDrop (difference between
AtP+QKfix and AtP∗) comes with a noticeable upfront cost and so looks worse in this comparison
while still helping avoid false negatives as shown inFigure 1.
3

<!-- page 4 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Model behaviour. Following past work (Chan et al., 2022; Geiger et al., 2022; Wang et al., 2022),
we assume a distributionD over pairs of inputs𝑥clean, 𝑥noise, where 𝑥clean is a prompt on which the
behaviour occurs, and𝑥noise is a reference prompt which we use as a source of noise to intervene
with1. We are also given a metric2 L : ℝ𝑉 → ℝ, which quantifies the behaviour of interest.
Contribution of a component. Similarly to the work referenced above we define the contribution
𝑐(𝑛) of a node𝑛 to the model’s behaviour as the counterfactual absolute3 expected impact of replacing
that node on the clean prompt with its value on the reference prompt𝑥noise.
Using do-calculus notation (Pearl, 2000) this can be expressed as𝑐(𝑛) := |I (𝑛)|, where
I (𝑛) := 𝔼(𝑥clean,𝑥noise)∼D
h
I (𝑛; 𝑥clean, 𝑥noise)
i
, (1)
where we define the intervention effectI for 𝑥clean, 𝑥noise as
I (𝑛; 𝑥clean, 𝑥noise) := L (M (𝑥clean | do(𝑛 ← 𝑛(𝑥noise)))) − L (M ( 𝑥clean)) . (2)
Note that the need to average the effect across a distribution adds a potentially large multiplicative
factor to the cost of computing𝑐(𝑛), further motivating this work.
We can also intervene on a set of nodes𝜂 = {𝑛𝑖}. To do so, we overwrite the values of all nodes in
𝜂 with their values from a reference prompt. Abusing notation, we write𝜂(𝑥) as the set of activations
of the nodes in𝜂, when computingM (𝑥).
I (𝜂; 𝑥clean, 𝑥noise) := L (M (𝑥clean | do(𝜂 ← 𝜂(𝑥noise)))) − L (M ( 𝑥clean)) (3)
We note that it is also valid to define contribution as the expected impact of replacing a node
on the reference prompt with its value on the clean prompt, also known as denoising or knock-in.
We follow Chan et al. (2022); Wang et al. (2022) in using noising, however denoising is also widely
used in the literature (Lieberum et al., 2023; Meng et al., 2023). We briefly consider how this choice
affects AtP in Section 5.2.
2.2. Attribution Patching
On state of the art models, computing𝑐(𝑛) for all𝑛 can be prohibitively expensive as there may be
billions or more nodes. Furthermore, to compute this value precisely requires evaluating it on all
prompt pairs, thus the runtime cost of Equation (1) for each𝑛 scales with the size of the support ofD.
We thus turn to a fast approximation of Equation (1). As suggested by Figurnov et al. (2016);
Molchanovetal.(2017);Nanda(2022),wecanmakeafirst-orderTaylorexpansionto I (𝑛; 𝑥clean, 𝑥noise)
around 𝑛(𝑥noise) ≈ 𝑛(𝑥clean):
1This precludes interventions which use activation values that are never actually realized, such as zero-ablation or mean
ablation. An alternative formulation via distributions of activation values is also possible.
2Common metrics in language models are next token prediction loss, difference in log prob between a correct and
incorrect next token, probability of the correct next token, etc.
3The sign of the impact may be of interest, but in this work we’ll focus on the magnitude, as a measure of causal
importance.
4

<!-- page 5 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
ˆIAtP(𝑛; 𝑥clean, 𝑥noise) := (𝑛(𝑥noise) − 𝑛(𝑥clean)) ⊺ 𝜕L (M (𝑥clean))
𝜕𝑛

𝑛=𝑛(𝑥clean )
(4)
Then, similarly to Syed et al. (2023), we apply this to a distribution by taking the absolute value
inside the expectation in Equation (1) rather than outside; this decreases the chance that estimates
across prompt pairs with positive and negative effects might erroneously lead to a significantly smaller
estimate. (We briefly explore the amount of cancellation behaviour in the true effect distribution in
Appendix B.2.) As a result, we get an estimate
ˆ𝑐AtP(𝑛) := 𝔼𝑥clean,𝑥noise
hˆIAtP(𝑛; 𝑥clean, 𝑥noise)

i
. (5)
This procedure is also calledAttribution Patching(Nanda, 2022) orAtP. AtP requires two forward
passes and one backward pass to compute an estimate score forall nodeson a given prompt pair, and
so provides a very significant speedup over brute force activation patching.
3. Methods
We now describe some failure modes of AtP and address them, yielding an improved method AtP*.
We then discuss some alternative methods for estimating𝑐(𝑛), to put AtP(*)’s performance in context.
Finally we discuss how to combine Subsampling, one such alternative method described in Section 3.3,
and AtP* to give a diagnostic to statistically test whether AtP* may have missed important false
negatives.
3.1. AtP improvements
We identify two common classes of false negatives occurring when using AtP.
The first failure mode occurs when the preactivation on𝑥clean is in a flat region of the activation
function (e.g. produces a saturated attention weight), but the preactivation on𝑥noise is not in that
region. As is apparent from Equation (4), AtP uses a linear approximation to the ground truth
in Equation (1), so if the non-linear function is badly approximated by the local gradient, AtP ceases
to be accurate – see Figure 3 for an illustration and Figure 4 which denotes in color the maximal
difference in attention observed between prompt pairs, suggesting that this failure mode occurs in
practice.
Another, unrelated failure mode occurs due to cancellation between direct and indirect effects:
roughly, if the total effect (on some prompt pair) is a sum of direct and indirect effects (Pearl,
2001) I (𝑛) = Idirect(𝑛) + I indirect(𝑛), and these are close to cancelling, then a small multiplicative
approximation error inˆIindirect
AtP (𝑛), due to non-linearities such as GELU and softmax, can accidentally
cause | ˆIdirect
AtP (𝑛) + ˆIindirect
AtP (𝑛)| to be orders of magnitude smaller than|I (𝑛)|.
3.1.1. False negatives from attention saturation
AtP relies on the gradient at each activation being reflective of the true behaviour of the function with
respect to intervention at that activation. In some cases, though, a node may immediately feed into a
non-linearity whose effect may not be adequately predicted by the gradient; for example, attention
key and query nodes feeding into the attention softmax non-linearity. To showcase this, we plot the
5

<!-- page 6 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure 3 | A linear approximation to the attention probability is a particularly poor approximation in
cases where one or both of the endpoints are in a saturated region of the softmax. Note that when
varying only a single key, the softmax becomes a sigmoid of the dot product of that key and the query.
true rank of each node’s effect against its rank assigned by AtP in Figure 4 (left). The plot shows
that there are many pronounced false negatives (below the dashed line), especially among keys and
queries.
Normal activation patching for queries and keys involves changing a query or key and then
re-running the rest of the model, keeping all else the same. AtP takes a linear approximation to the
entire rest of the model rather than re-running it. We propose explicitly re-computing the first step of
the rest of the model, i.e. the attention softmax, and then taking a linear approximation to the rest.
Formally, for attention key and query nodes, instead of using the gradient on those nodes directly,
we take the difference in attention weight caused by that key or query, multiplied by the gradient
on the attention weights themselves. This requires finding the change in attention weights from
each key and query patch — but that can be done efficiently using (for all keys and queries in total)
less compute than two transformer forward passes. This correction avoids the problem of saturated
attention, while otherwise retaining the performance of AtP.
Queries For the queries, we can easily compute the adjusted effect by running the model on𝑥noise
and caching the noise queries. We then run the model on𝑥clean and cache the attention keys and
weights. Finally, we compute the attention weights that result from combining all the keys from the
𝑥clean forward pass with the queries from the𝑥noise forward pass. This costs approximately as much as
the unperturbed attention computation of the transformer forward pass. For each query node𝑛 we
refer to the resulting weight vector asattn(𝑛)patch, in contrast with the weightsattn(𝑛) (𝑥clean) from
the clean forward pass. The improved attribution estimate for𝑛 is then
6

<!-- page 7 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
ˆI𝑄
AtPfix(𝑛; 𝑥clean, 𝑥noise) :=
∑︁
𝑘
ˆIAtP(attn(𝑛)𝑘; 𝑥clean, 𝑥noise) (6)
= (attn(𝑛)patch −attn(𝑛) (𝑥clean)) ⊺ 𝜕L (M (𝑥clean))
𝜕attn(𝑛)

attn(𝑛)=attn(𝑛) (𝑥clean )
(7)
Keys For the keys we first describe a simple but inefficient method. We again run the model on𝑥noise,
caching the noise keys. We also run it on𝑥clean, caching the clean queries and attention probabilities.
Let key nodes for a single attention head be𝑛𝑘
1, . . . , 𝑛𝑘
𝑇 and letqueries(𝑛𝑘
𝑡 ) = {𝑛𝑞
1, . . . , 𝑛𝑞
𝑇 } be the set of
query nodes for the same head as node𝑛𝑘
𝑡. We then define
attn𝑡
patch(𝑛𝑞) := attn(𝑛𝑞) (𝑥clean | do(𝑛𝑘
𝑡 ← 𝑛𝑘
𝑡 (𝑥noise))) (8)
Δ𝑡 attn(𝑛𝑞) := attn𝑡
patch(𝑛𝑞) − attn(𝑛𝑞) (𝑥clean) (9)
The improved attribution estimate for𝑛𝑘
𝑡 is then
ˆI 𝐾
AtPfix(𝑛𝑘
𝑡; 𝑥clean, 𝑥noise) :=
∑︁
𝑛𝑞 ∈queries(𝑛𝑘
𝑡 )
Δ𝑡 attn(𝑛𝑞)⊺ 𝜕L (M (𝑥clean))
𝜕attn(𝑛𝑞)

attn(𝑛𝑞 )=attn(𝑛𝑞 ) (𝑥clean )
(10)
However, the procedure we just described is costly to execute as it requiresO(𝑇3) flops to naively
compute Equation (9) for all𝑇 keys. In Appendix A.2.1 we describe a more efficient variant that
takes no more compute than the forward pass attention computation itself (requiringO(𝑇2) flops).
Since Equation (6) is also cheaper to compute than a forward pass, the full QK fix requires less than
two transformer forward passes (since the latter also includes MLP computations).
For attention nodes we show the effects of applying the query and key fixes in Figure 4 (middle).
We observe that the propagation of Q/K effects has a major impact on reducing the false negative
rate.
3.1.2. False negatives from cancellation
This form of cancellation occurs when the backpropagated gradient from indirect effects is combined
with the gradient from the direct effect. We propose a way to modify the backpropagation within the
attribution patching to reduce this issue. If we artificially zero out the gradient at a downstream layer
that contributes to the indirect effect, the cancellation is disrupted. (This is also equivalent to patching
in clean activations at the outputs of the layer.) Thus we propose to do this iteratively, sweeping
across the layers. Any node whose effect does not route through the layer being gradient-zeroed will
have its estimate unaffected.
We call this methodGradDrop. For every layerℓ ∈ {1, . . . , 𝐿} in the model, GradDrop computes an
AtP estimate for all nodes, where gradients on the residual contribution fromℓ are set to 0, including
the propagation to earlier layers. This provides a different estimate for all nodes, for each layer that
was dropped. We call the so-modified gradient𝜕Lℓ
𝜕𝑛 = 𝜕L
𝜕𝑛 (M (𝑥clean | do(𝑛out
ℓ ← 𝑛out
ℓ (𝑥clean)))) when
dropping layerℓ, where𝑛out
ℓ is the contribution to the residual stream across all positions. Using𝜕Lℓ
𝜕𝑛 in
place of 𝜕Lℓ
𝜕𝑛 in the AtP formula produces an estimateˆIAtP+GDℓ (𝑛). Then, the estimates are aggregated
7

<!-- page 8 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure 4 | Ranks of 𝑐(𝑛) against ranks ofˆ𝑐AtP(𝑛), on Pythia-12B onCITY-PP. Both improvements
to AtP reduce the number of false negatives (bottom right triangle area), where in this case most
improvements come from the QK fix. Coloration indicates the maximum absolute difference in
attention probability when comparing𝑥clean and patching a given query or key. Many false negatives
are keys and queries with significant maximum difference in attention probability, suggesting they
are due to attention saturation as illustrated in Figure 3. Output and value nodes are colored in grey
as they do not contribute to the attention probability.
by averaging their absolute values, and then scaling by𝐿
𝐿−1 to avoid changing the direct-effect path’s
contribution (which is otherwise zeroed out when dropping the layer the node is in).
ˆ𝑐AtP+GD(𝑛) := 𝔼𝑥clean,𝑥noise
"
1
𝐿 −1
𝐿∑︁
ℓ=1
ˆIAtP+GDℓ (𝑛; 𝑥clean, 𝑥noise)

#
(11)
Notethattheforwardpassesrequiredforcomputing ˆIAtP+GDℓ (𝑛; 𝑥clean, 𝑥noise) don’tdependonℓ, so
the extra compute needed for GradDrop is𝐿 backwards passes from the same intermediate activations
on a clean forward pass. This is also the case with the QK fix: the corrected attributionsˆIAtPfix are
dot products with the attention weight gradients, so the only thing that needs to be recomputed for
ˆIAtPfix+GDℓ (𝑛) is the modified gradient 𝜕Lℓ
𝜕attn(𝑛). Thus, computing Equation (11) takes𝐿 backwards
passes4 on top of the costs for AtP.
We show the result of applying GradDrop on attention nodes in Figure 4 (right) and on MLP nodes
in Figure 5. In Figure 5, we show the true effect magnitude rank against the AtP+GradDrop rank,
while highlighting nodes which improved drastically by applying GradDrop. We give some arguments
and intuitions on the benefit of GradDrop in Appendix A.2.2.
Direct Effect Ratio To provide some evidence that the observed false negatives are due to cancella-
tion, we compute the ratio between the direct effect𝑐direct(𝑛) and the total effect𝑐(𝑛). A higher direct
effect ratio indicates more cancellation. We observe that the most significant false negatives corrected
by GradDrop in Figure 5 (highlighted) have high direct effect ratios of5.35, 12.2, and0 (no direct
effect) , while the median direct effect ratio of all nodes is0 (if counting all nodes) or0.77 (if only
counting nodes that have direct effect). Note that direct effect ratio is only applicable to nodes which
4This can be reduced to(𝐿 +1)/2 by reusing intermediate results.
8

<!-- page 9 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure 5 | True rank and rank of AtP estimates with and without GradDrop, using Pythia-12B on
the CITY-PP distribution with NeuronNodes. GradDrop provides a significant improvement to the
largest neuron false negatives (red circles) relative to Default AtP (orange crosses).
in fact have a direct connection to the output, and not e.g. to MLP nodes at non-final token positions,
since all disconnected nodes have a direct effect of 0 by definition.
3.2. Diagnostics
Despite the improvements we have proposed in Section 3.1, there is no guarantee that AtP* produces
no false negatives. Thus, it is desirable to obtain an upper confidence bound on the effect size of
nodes that might be missed by AtP*, i.e. that aren’t in the top𝐾 AtP* estimates, for some𝐾. Let the
top 𝐾 nodes be Top𝐾
𝐴𝑡𝑃 ∗. It so happens that we can use subset sampling to obtain such a bound.
As described in Algorithm 1 and Section 3.3, the subset sampling algorithm returns summary
statistics: ¯𝑖𝑛
±, 𝑠𝑛
± and count𝑛
± for each node𝑛: the average effect size¯𝑖𝑛
± of a subset conditional on the
node being contained in that subset (+) or not (−), the sample standard deviations𝑠𝑛
±, and the sample
sizes count𝑛
±. Given these, consider a null hypothesis5 𝐻𝑛
0 that |I (𝑛)| ≥ 𝜃, for some threshold𝜃, versus
the alternative hypothesis𝐻𝑛
1 that |I (𝑛)| < 𝜃. We use a one-sided Welch’s t-test6 to test this hypothesis;
the general practice with a compound null hypothesis is to select the simple sub-hypothesis that gives
the greatest 𝑝-value, so to be conservative, the simple null hypothesis is thatI (𝑛) = 𝜃sign(¯𝑖𝑛
+ −¯𝑖𝑛
−),
giving a test statistic of𝑡𝑛 = (𝜃 − |¯𝑖𝑛
+ −¯𝑖𝑛
− |)/ 𝑠𝑛
Welch, which gives a𝑝-value of 𝑝𝑛 = ℙ𝑇∼𝑡𝜈𝑛
Welch
(𝑇 > 𝑡 𝑛).
To get a combined conclusion across all nodes in𝑁 \ Top𝐾
𝐴𝑡𝑃 ∗, let’s consider the hypothesis
𝐻0 = Ô
𝑛∈ 𝑁\Top𝐾
𝐴𝑡𝑃 ∗
𝐻𝑛
0 that any of those nodes has true effect|I (𝑛)| > 𝜃. Since this is also a compound
null hypothesis,max𝑛 𝑝𝑛 is the corresponding𝑝-value. Then, to find an upper confidence bound with
5This is an unconventional form of𝐻0 – typically a null hypothesis will say that an effect is insignificant. However, the
framework of statistical hypothesis testing is based on determining whether the data let us reject the null hypothesis, and
in this case the hypothesis we want to reject is the presence, rather than the absence, of a significant false negative.
6This relies on the populations being approximately unbiased and normally distributed, and not skewed. This tended to
be true on inspection, and it’s what the additivity assumption (see Section 3.3) predicts for a single prompt pair — but a
nonparametric bootstrap test may be more reliable, at the cost of additional compute.
9

<!-- page 10 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
(a) IOI-PP
 (b) IOI
Figure 6 | Upper confidence bounds on effect magnitudes of false negatives (i.e. nodes not in the top
1024 nodes according to AtP∗), at 3 confidence levels, varying the sampling budget. On the left we
show in red the true effect of the nodes which are ranked highest by AtP∗. We also show the true
effect magnitude at various ranks of the remaining nodes in orange.
We can see that the bound for (a) finds the true biggest false negative reasonably early, while for (b),
where there is no large false negative, we progressively keep gaining confidence with more data.
Note that the costs involved per prompt pair are substantially different between the subplots, and in
particular this diagnostic for the distributional case (b) is substantially cheaper to compute than the
verification cost of 1024 samples per prompt pair.
specified confidence level1 − 𝑝, we invert this procedure to find the lowest𝜃 for which we still have at
least that level of confidence. We repeat this for various settings of the sample size𝑚 in Algorithm 1.
The exact algorithm is described in Appendix A.3.
In Figure 6, we report the upper confidence bounds at confidence levels 90%, 99%, 99.9% from
running Algorithm 1 with a given𝑚 (right subplots), as well as the number of nodes that have a true
contribution 𝑐(𝑛) greater than 𝜃 (left subplots).
3.3. Baselines
Iterative The most straightforward method is to directly do Activation Patching to find the true
effect 𝑐(𝑛) of each node, in some uninformed random order. This is necessarily inefficient.
However, if we are scaling to a distribution, it is possible to improve on this, by alternating between
phases of (i) for each unverified node, picking a not-yet-measured prompt pair on which to patch
it, (ii) ranking the not-yet-verified nodes by the average observed patch effect magnitudes, taking
the top |𝑁 |/|D | nodes, and verifying them. This balances the computational expenditure on the two
tasks, and allows us to find large nodes sooner, at least as long as their large effect shows up on many
prompt pairs.
Our remaining baseline methods rely on an approximatenode additivity assumption: that when
intervening on a set of nodes𝜂, the measured effectI (𝜂; 𝑥clean, 𝑥noise) is approximately equal toÍ
𝑛∈𝜂 I (𝑛; 𝑥clean, 𝑥noise).
Subsampling Under the approximate node additivity assumption, we can construct an approxi-
mately unbiased estimator of𝑐(𝑛). We select the sets𝜂𝑘 to contain each node independently with
10

<!-- page 11 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
some probability 𝑝, and additionally sample prompt pairs𝑥clean
𝑘 , 𝑥noise
𝑘 ∼ D. For any node𝑛, and sets
of nodes𝜂𝑘 ⊂ 𝑁, let 𝜂+(𝑛) be the collection of all those that contain𝑛, and 𝜂− (𝑛) be the collection
of those that don’t contain𝑛; we’ll write these node sets as𝜂+
𝑘 (𝑛) and 𝜂−
𝑘 (𝑛), and the corresponding
prompt pairs as𝑥clean
𝑘
+
(𝑛), 𝑥noise
𝑘
+
(𝑛) and 𝑥clean
𝑘
−
(𝑛), 𝑥noise
𝑘
−
(𝑛). The subsampling (or subset sampling)
estimator is then given by
ˆISS (𝑛) := 1
|𝜂+(𝑛)|
|𝜂+ (𝑛) |∑︁
𝑘=1
I (𝜂+
𝑘 (𝑛); 𝑥clean
𝑘
+
(𝑛), 𝑥noise
𝑘
+
(𝑛)) − 1
|𝜂− (𝑛)|
|𝜂− (𝑛) |∑︁
𝑘=1
I (𝜂−
𝑘 (𝑛); 𝑥clean
𝑘
−
(𝑛), 𝑥noise
𝑘
−
(𝑛))
(12)
ˆ𝑐SS (𝑛) := | ˆISS (𝑛)| (13)
Theestimator ˆISS (𝑛) isunbiasediftherearenointeractioneffects, andhasasmallbiasproportional
to 𝑝 under a simple interaction model (see Appendix A.1.1 for proof).
In practice, we compute all the estimatesˆ𝑐SS (𝑛) by sampling a binary mask over all nodes from
i.i.d. Bernoulli| 𝑁 | ( 𝑝) – each binary mask can be identified with a node set𝜂. In Algorithm 1, we
describe how to compute summary statistics related to Equation (13) efficiently for all nodes𝑛 ∈ 𝑁.
The means¯𝑖± are enough to computeˆ𝑐SS (𝑛), while other summary statistics are involved in bounding
the magnitude of a false negative (cf. Section 3.2). (Note,count±
𝑛 is just an alternate notation for
|𝜂±(𝑛)|.)
Algorithm 1Subsampling
Require: 𝑝 ∈ (0,1), modelM, metric L, prompt pair distributionD, num samples𝑚
1: count±, runSum±, runSquaredSum± ← 0| 𝑁 | ⊲ Init counts and running sums to 0 vectors
2: for 𝑖 ← 1 to 𝑚 do
3: 𝑥clean, 𝑥noise ∼ D
4: mask+ ← Bernoulli| 𝑁 | ( 𝑝) ⊲ Sample binary mask for patching
5: mask− ← 1 −mask+
6: 𝑖 ← I ({𝑛 ∈ 𝑁 : mask+
𝑛 = 1}; 𝑥clean, 𝑥noise) ⊲ 𝜂 + = {𝑛 ∈ 𝑁 : mask+
𝑛 = 1}
7: count± ← count± +mask±
8: runSum± ← runSum± + 𝑖 ·mask±
9: runSquaredSum± ← runSquaredSum± + 𝑖2 ·mask±
10: ¯𝑖± ← runSum±/count±
11: 𝑠± ←
√︁
(runSquaredSum± − (¯𝑖±)2)/(count± −1)
12: return count±, ¯𝑖±, 𝑠± ⊲ If diagnostics are not required,¯𝑖± is sufficient.
Blocks & Hierarchical Instead of sampling each𝜂 independently, we can group nodes into fixed
“blocks”𝜂 of some size, and patch each block to find its aggregated contribution𝑐(𝜂); we can then
traverse the nodes, starting with high-contribution blocks and proceeding from there.
There is a tradeoff in terms of the block size: using large blocks increases the compute required to
traverse a high-contribution block, but using small blocks increases the compute required to finish
traversing all of the blocks. We refer to the fixed block size setting asBlocks. Another way to handle
this tradeoff is to add recursion: the blocks can be grouped into higher-level blocks, and so forth. We
call this methodHierarchical.
We present results from both methods in our comparison plots, but relegate details to Ap-
pendix A.1.2. Relative to subsampling, these grouping-based methods have the disadvantage that on
11

<!-- page 12 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
distributions, their cost scales linearly with size ofD’s support, in addition to scaling with the number
of nodes7.
4. Experiments
4.1. Setup
Nodes When attributing model behavior to components, an important choice is the partition of the
model’s computational graph into units of analysis or ‘nodes’𝑁 ∋ 𝑛 (cf. Section 2.1). We investigate
two settings for the choice of𝑁, AttentionNodes and NeuronNodes. For NeuronNodes, each MLP
neuron8 is a separate node. ForAttentionNodes, we consider the query, key, and value vector for
each head as distinct nodes, as well as the pre-linear per-head attention output9. We also refer to
these units as ‘sites’. For each site, we consider each copy of that site at different token positions as
a separate node. As a result, we can identify each node𝑛 ∈ 𝑁 with a pair(𝑇, 𝑆) from the product
TokenPosition × Site. Since our two settings for𝑁 are using a different level of granularity and are
expected to have different per-node effect magnitudes, we present results on them separately.
Models We investigate transformer language models from the Pythia suite (Biderman et al., 2023)
of sizes between 410M and 12B parameters. This allows us to demonstrate that our methods are
applicable across scale. Our cost-of-verified-recall plots in Figures 1, 7 and 8 refer to Pythia-12B.
Results for other model sizes are presented via the relative-cost (cf. Section 4.2) plots in the main
body Figure 9 and disaggregated via cost-of-verified recall in Appendix B.3.
Effect Metric L All reported results use the negative log probability10 as their loss functionL.
We computeL relative to targets from the clean prompt𝑥clean. We briefly explore other metrics in
Appendix B.4.
4.2. Measuring Effectiveness and Efficiency
Cost of verified recall As mentioned in the introduction, we’re primarily interested in finding the
largest-effect nodes – see Appendix D for the distribution of𝑐(𝑛) across models and distributions.
Once we have obtained node estimates via a given method, it is relatively cheap to directly measure
true effects of top nodes one at a time; we refer to this as “verification”. Incorporating this into our
methodology, we find that false positives are typically not a big issue; they are simply revealed during
verification. In contrast, false negatives are not so easy to remedy without verifying all nodes, which
is what we were trying to avoid.
We compare methods on the basis of total compute cost (in # of forward passes) to verify the𝐾
nodes with biggest true effect magnitude, for varying𝐾. The procedure being measured is to first
compute estimates (incurring an estimation cost), and then sweep through nodes in decreasing order
7AtP* also scales linearly in the same way, but with far fewer forward passes per prompt pair.
8We use the neuron post-activation for the node; this makes no difference when causally intervening, but for AtP it’s
beneficial, because it makes the𝑛 ↦→ L (𝑛) function more linear.
9We include the output node because it provides additional information about what function an attention head is serving,
particularly in the case where its queries have negligible patch effects relative to its keys and/or values. This may happen as
a result of choosing𝑥clean, 𝑥 noise such that the query does not differ across the prompts.
10Another popular metric is the difference in logits between the clean and noise target. As opposed to the negative
logprob, the logit difference is linear in the final logits and thus might favor AtP. A downside of logit difference is that it is
sensitive to the noise target, which may not be meaningful if there are multiple plausible completions, such as inIOI.
12

<!-- page 13 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
of estimated magnitude, measuring their individual effects𝑐(𝑛) (i.e. verifying them), and incurring a
verification cost. Then the total cost is the sum of these two costs.
Inverse-rank-weighted geometric mean cost Sometimes we find it useful to summarize the
method performance with a scalar; this is useful for comparing methods at a glance across different
settings (e.g. model sizes, as in Figure 2), or for selecting hyperparameters (cf. Appendix B.5). The
cost of verified recall of the top𝐾 nodes is of interest for𝐾 at varying orders of magnitude. In order
to avoid the performance metric being dominated by small or large𝐾, we assign similar total weight
to different orders of magnitude: we use a weighted average with weight1/𝐾 for the cost of the top
𝐾 nodes. Similarly, since the costs themselves may have different orders of magnitude, we average
them on a log scale – i.e., we take a geometric mean.
This metric is also proportional to the area under the curve in plots like Figure 1. To produce a
more understandable result, we always report it relative to (i.e. divided by) the oracle verification
cost on the same metric; the diagonal line is the oracle, with relative cost 1. We refer to this as the
IRWRGM (inverse-rank-weighted relative geometric mean) cost, or the relative cost.
Note that the preference of the individual practitioner may be different such that this metric is no
longer accurately measuring the important rank regime. For example, AtP* pays a notable upfront
cost relative to AtP or AtP+QKfix, which sets it at a disadvantage when it doesn’t manage to find
additional false negatives; but this may or may not be practically significant. To understand the
performance in more detail we advise to refer to the cost of verified recall plots, like Figure 1 (or
many more in Appendix B.3).
4.3. Single Prompt Pairs versus Distributions
We focus many of our experiments on single prompt pairs. This is primarily because it’s easier to set
up and get ground truth data. It’s also a simpler setting in which to investigate the question, and one
that’s more universally applicable, since a distribution to generalize to is not always available.
Clean single prompt pairs As a starting point we report results on single prompt pairs which we
expect to have relatively clean circuitry11. All singular prompt pairs are shown in Table 1.IOI-PP is
chosen to resemble an instance from the indirect object identification (IOI) task (Wang et al., 2022),
a task predominantly involving attention heads.CITY-PP is chosen to elicit factual recall which
previous research suggests involves early MLPs and a small number of late attention heads (Geva
et al., 2023; Meng et al., 2023; Nanda et al., 2023). The country/city combinations were chosen such
that Pythia-410M achieved low loss on both𝑥clean and 𝑥noise and such that all places were represented
by a single token.
We show the cost of verified 100% recall for various methods in Figure 1, where we focus on
NeuronNodes forCITY-PP and AttentionNodes forIOI-PP. Exhaustive results for smaller Pythia
models are shown in Appendix B.3. Figure 2 shows the aggregated relative costs for all models on
CITY-PP and IOI-PP.
Instead of applying the strict criterion of recalling all important nodes, we can also relax this
constraint. In Figure 7, we show the cost of verified 90% recall in the two clean prompt pair settings.
11Formally, these represent prompt distributions via the delta distribution𝑝(𝑥clean, 𝑥noise) = 𝛿𝑥clean
1 ,𝑥noise
1
(𝑥clean, 𝑥noise)
where 𝑥clean
1 , 𝑥noise
1 is the singular prompt pair.
13

<!-- page 14 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
(a) NeuronNodes onCITY-PP
 (b) AttentionNodes onIOI-PP
Figure 7 | Costs of finding the most causally-important nodes in Pythia-12B using different methods
on clean prompt pairs, with 90% target recall. This highlights that the AtP* false negatives in Figure 1
are a small minority of nodes.
Random prompt pair The previous prompt pairs may in fact be the best-case scenarios: the
interventions they create will be fairly localized to a specific circuit, and this may make it easy for AtP
to approximate the contributions. It may thus be informative to see how the methods generalize to
settings where the interventions are less surgical. To do this, we also report results in Figure 8 (top)
and Figure 9 on a random prompt pair chosen from a non-copyright-protected section of The Pile (Gao
et al., 2020) which we refer to asRAND-PP. The prompt pair was chosen such that Pythia-410M still
achieved low loss on both prompts.
We find that AtP/AtP* is only somewhat less effective here; this provides tentative evidence that
the strong performance of AtP/AtP* isn’t reliant on the clean prompt using a particularly crisp circuit,
or on the noise prompt being a precise control.
Distributions Causal attribution is often of most interest when evaluated across a distribution, as laid
out in Section 2. Of the methods, AtP, AtP*, and Subsampling scale reasonably to distributions; the
former 2 because they’re inexpensive so running them|D | times is not prohibitive, and Subsampling
because it intrinsically averages across the distribution and thus becomes proportionally cheaper
relative to the verification via activation patching. In addition, having a distribution enables a more
performant Iterative method, as described in Section 3.3.
We present a comparison of these methods on 2 distributional settings. The first is a reduced
version of IOI (Wang et al., 2022) on 6 names, resulting in6 ×5 ×4 = 120 prompt pairs, where we
evaluate AttentionNodes. The other distribution prompts the model to output an indefinite article
‘a’ or ‘an’, where we evaluateNeuronNodes. See Appendix B.1 for details on constructing these
distributions. Results are shown in Figure 8 for Pythia 12B, and in Figure 9 across models. The results
show that AtP continues to perform well, especially with the QK fix; in addition, the cancellation
failure mode tends to be sensitive to the particular input prompt pair, and as a result, averaging across
14

<!-- page 15 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
(a) RAND-PP MLP neurons.
 (b) RAND-PP Attention nodes.
(c) A-AN MLP neurons.
 (d) IOI Attention nodes.
Figure 8 | Costs of finding the most causally-important nodes in Pythia-12B using different methods,
on a random prompt pair (see Table 1) and on distributions. The shading indicates geometric standard
deviation. Cost is measured in forward passes, or forward passes per prompt pair in the distributional
case.
15

<!-- page 16 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
(a) RAND-PP MLP neurons.
 (b) RAND-PP Attention nodes.
(c) A-AN MLP neurons.
 (d) IOI Attention nodes.
Figure 9 | Costs of methods across models, on random prompt pair and on distributions. The costs are
relative to having an oracle (and thus verifying nodes in decreasing order of true contribution size);
they’re aggregated using an inverse-rank-weighted geometric mean. This means they correspond to
the area above the diagonal for each curve in Figure 8.
16

<!-- page 17 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Identifier Clean Prompt Noise Source Prompt
CITY-PP
BOS City : ␣Barcelona\n
Country : ␣Spain
BOS City : ␣Beijing\n
Country : ␣China
IOI-PP
BOS When ␣Michael␣and␣Jessica
␣went␣to␣the␣bar, ␣Michael
␣gave␣a␣drink␣to ␣Jessica
BOS When ␣Michael␣and␣Jessica
␣went␣to␣the␣bar, ␣Ashley
␣gave␣a␣drink␣to␣Michael
RAND-PP
BOS Her ␣biggest␣worry␣was␣the
␣festival␣might␣suffer␣and
␣people␣might␣erroneously ␣think
BOS also ␣think␣that␣there
␣should␣be␣the␣same␣rules
␣or␣regulations␣when␣it
Table 1| Clean and noise source prompts for singular prompt pair distributions. Vertical lines denote
tokenization boundaries. All prompts are preceded by the BOS (beginning of sequence) token. The
last token is not part of the input. The last token of the clean prompt is used as the target inL.
a distribution diminishes the benefit of GradDrops.
An implication of Subsampling scaling well to this setting is that diagnostics may give reasonable
confidence in not missing false negatives with much less overhead than in the single-prompt-pair
case; this is illustrated in Figure 6.
5. Discussion
5.1. Limitations
Prompt pair distributions We only considered a small set of prompt pair distributions, which often
were limited to a single prompt pair, since evaluating the ground truth can be quite costly. While we
aimed to evaluate on distributions that are reasonably representative, our results may not generalize
to other distributions.
Choice of Nodes𝑁 In the NeuronNodes setting, we took MLP neurons as our fundamental unit of
analysis. However, there is mounting evidence (Bricken et al., 2023) that the decomposition of signals
into neuron contributions does not correspond directly to a semantically meaningful decomposition.
Instead, achieving such a decomposition seems to require finding the right set of directions in neuron
activation space (Bricken et al., 2023; Gurnee et al., 2023) – which we viewed as being out of scope
for this paper. In Section 5.2 we further discuss the applicability of AtP to sparse autoencoders, a
method of finding these decompositions.
More generally, we only considered relatively fine-grained nodes, because this is a case where
very exhaustive verification is prohibitively expensive, justifying the need for an approximate, fast
method. Nanda (2022) speculate that AtP may perform worse on coarser components like full layers
or entire residual streams, as a larger change may have more of a non-linear effect. There may still be
benefit in speeding up such an analysis, particularly if the context length is long – our alternative
methods may have something to offer here, though we leave investigation of this to future work.
It is popular in the literature to do Activation Patching with these larger components, with short
17

<!-- page 18 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
contexts – this doesn’t pose a performance issue, and so our work would not provide any benefit here.
Caveats of 𝑐(𝑛) as importance measure In this work we took the ground truth of activation
patching, as defined in Equation (1), as our evaluation target. As discussed by McGrath et al. (2023),
Equation (1) often significantly disagrees with a different evaluation target, the “direct effect”, by
putting lower weight on some contributions when later components would shift their behaviour to
compensate for the earlier patched component. In the worst case this could be seen as producing
additional false negatives not accounted for by our metrics. To some degree this is likely to be
mitigated by the GradDrop formula in Eq. (11), which will include a term dropping out the effect of
that downstream shift.
However, it is also questionable whether we need to concern ourselves with finding high-direct-
effect nodes. For example, direct effect is easy to efficiently compute for all nodes, as explored by
nostalgebraist (2020) – so there is no need for fast approximations like AtP if direct effect is the
quantity of interest. This ease of computation is no free lunch, though, because direct effect is also
more limited as a tool for finding causally important nodes: it would not be able to locate any nodes
that contribute only instrumentally to the circuit rather than producing its output. For example, there
is no direct effect from nodes at non-final token positions. We discuss the direct effect further in
Section 3.1.2 and Appendix A.2.2.
Another nuance of our ground–truth definition occurs in the distributional setting. Some nodes
may have a real and significant effect, but only on a single clean prompt (e.g. they only respond
to a particular name inIOI12 or object inA-AN). Since the effect is averaged over the distribution,
the ground truth will not assign these nodes large causal importance. Depending on the goal of the
practitioner this may or may not be desirable.
Effect size versus rank estimationWhen evaluating the performance of various estimators, we
focused on evaluating the relative rank of estimates, since our main goal was to identify important
components (with effect size only instrumentally useful to this end), and we assumed a further
verification step of the nodes with highest estimated effects one at a time, in contexts where knowing
effect size is important. Thus, we do not present evidence about how closely the estimated effect
magnitudes from AtP or AtP* match the ground truth. Similarly, we did not assess the prevalence of
false positives in our analysis, because they can be filtered out via the verification process. Finally, we
did not compare to past manual interpretability work to check whether our methods find the same
nodes to be causally important as discovered by human researchers, as done in prior work (Conmy
et al., 2023; Syed et al., 2023).
Other LLMs While we think it likely that our results on the Pythia model family (Biderman et al.,
2023) will transfer to other LLM families, we cannot rule out qualitatively different behavior without
further evidence, especially on SotA–scale models or models that significantly deviate from the
standard decoder-only transformer architecture.
5.2. Extensions/Variants
EdgePatching Whilewefocusoncomputingtheeffectsofindividualnodes, edgeactivationpatching
cangivemorefine-grainedinformationaboutwhichpathsinthecomputationalgraphmatter. However,
it suffers from an even larger blowup in number of forward passes if done naively. Fortunately, AtP is
12We did observe this particular behavior in a few instances.
18

<!-- page 19 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
easy to generalize to estimating the effects of edges between nodes (Nanda, 2022; Syed et al., 2023),
while AtP* may provide further improvement. We discuss edge-AtP, and how to efficiently carry over
the insights from AtP*, in Appendix C.2.
Coarser nodes 𝑁 We focused on fine-grained attribution, rather than full layers or sliding win-
dows (Geva et al., 2023; Meng et al., 2023). In the latter case there’s less computational blowup to
resolve, but for long contexts there may still be benefit in considering speedups like ours; on the other
hand, they may be less linear, thus favouring other methods over AtP*. We leave investigation of this
to future work.
Layer normalization Nanda (2022) observed that AtP’s approximation to layer normalization may
be a worse approximation when it comes to patching larger/coarser nodes: on average the patched
and clean activations are likely to have similar norm, but may not have high cosine-similarity. They
recommend treating the denominator in layer normalization as fixed, e.g. using a stop-gradient
operator in the implementation. In Appendix C.1 we explore the effect of this, and illustrate the
behaviour of this alternative form of AtP. It seems likely that this variant would indeed produce better
results particularly when patching residual-stream nodes – but we leave empirical investigation of
this to future work.
Denoising Denoising (Lieberum et al., 2023; Meng et al., 2023) is a different use case for patching,
which may produce moderately different results: the difference is that each forward pass is run on
𝑥noise with the activation to patch taken from𝑥clean — colloquially, this tests whether the patched
activation is sufficient to recover model performance on𝑥clean, rather than necessary. We provide
some preliminary evidence to the effect of this choice in Appendix B.4 but leave a more thorough
investigation to future work.
Other forms of ablation Further, in some settings it may be of interest to do mean-ablation, or
even zero-ablation, and our tweaks remain applicable there; the random-prompt-pair result suggests
AtP* isn’t overly sensitive to the noise distribution, so we speculate the results are likely to carry over.
5.3. Applications
Automated Circuit Finding A natural application of the methods we discussed in this work is the
automatic identification and localization of sparse subgraphs or ‘circuits’ (Cammarata et al., 2020). A
variant of this was already discussed in concurrent work by Syed et al. (2023) who combined edge
attribution patching with the ACDC algorithm (Conmy et al., 2023). As we mentioned in the edge
patching discussion, AtP* can be generalized to edge attribution patching, which may bring additional
benefit for automated circuit discovery.
Anotherapproachistolearna(probabilistic)maskovernodes, similartoCaoetal.(2021);Louizos
et al. (2018), where the probability scales with the currently estimated node contribution𝑐(𝑛). For
that approach, a fast method to estimate all node effects given the current mask probabilities could
prove vital.
Sparse Autoencoders Recently there has been increased interest by the community in using sparse
autoencoders (SAEs) to construct disentangled sparse representations with potentially more semantic
coherence than transformer-native units such as neurons (Bricken et al., 2023; Cunningham et al.,
19

<!-- page 20 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
2023). SAEs usually have a lot more nodes than the corresponding transformer block they are applied
to. This could pose a larger problem in terms of the activation patching effects, making the speedup
of AtP* more valuable. However, due to the sparseness of the SAE, on a given forward pass the
effect of most features will be zero. For example, some successful SAEs by Bricken et al. (2023)
have 10-20 active features for 500 neurons for a given token position, which reduces the number of
nodes by 20-50x relative to the MLP setting, increasing the scale at which existing iterative methods
remain practical. It is still an open research question, however, what degree of sparsity is feasible
with tolerable reconstruction error for practically relevant or SOTA–scale models, where the methods
discussed in this work may become more important again.
Steering LLMs AtP* could be used to discover single nodes in the model that can be leveraged for
targeted inference time interventions to control the model’s behavior. In contrast to previous work (Li
et al., 2023; Turner et al., 2023; Zou et al., 2023) it might provide more localized interventions with
less impact on the rest of the model’s computation. One potential exciting direction would be to use
AtP* (or other gradient-based approximations) to see which sparse autoencoder features, if activated,
would have a significant effect.
5.4. Recommendation
Our results suggest that if a practitioner is trying to do fast causal attribution, there are 2 main factors
to consider: (i) the desired granularity of localization, and (ii) the confidence vs compute tradeoff.
Regarding (i), the desired granularity, smaller components (e.g. MLP neurons or attention heads)
are more numerous but more linear, likely yielding better results from gradient-based methods like
AtP. We are less sure AtP will be a good approximation if patching layers or sliding windows of layers,
and in this case practitioners may want to do normal patching. If the number of forward passes
required remains prohibitive (e.g. a long context times many layers, when doing per token× layer
patching), our other baselines may be useful. For a single prompt pair we particularly recommend
trying Blocks, as it’s easy to make sense of; for a distribution we recommend Subsampling because it
scales better to many prompt pairs.
Regarding (ii), the confidence vs compute tradeoff, depending on the application, it may be
desirable to run AtP as an activation patching prefilter followed by running the diagnostic to increase
confidence. On the other hand, if false negatives aren’t a big concern then it may be preferable to
skip the diagnostic – and if false positives aren’t either, then in certain cases practitioners may want
to skip activation patching verification entirely. In addition, if the prompt pair distribution does not
adequately highlight the specific circuit/behaviour of interest, this may also limit what can be learned
from any localization methods.
If AtP is appropriate, our results suggest the best variant to use is probably AtP* for single prompt
pairs, AtP+QKFix forAttentionNodes on distributions, and AtP forNeuronNodes (or other sites that
aren’t immediately before a nonlinearity) on distributions.
Of course, these recommendations are best-substantiated in settings similar to those we studied:
focused prompt pairs / distribution, attention node or neuron sites, nodewise attribution, measuring
cross-entropylossontheclean-promptnexttoken. Ifdepartingfromtheseassumptionswerecommend
looking before you leap.
20

<!-- page 21 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
6. Related work
Localization and Mediation Analysis This work is concerned with identifying the effect of all
(important) nodes in a causal graph (Pearl, 2000), in the specific case where the graph represents a
language model’s computation. A key method for finding important intermediate nodes in a causal
graph is intervening on those nodes and observing the effect, which was first discussed under the
name of causal mediation analysis by Pearl (2001); Robins and Greenland (1992).
Activation Patching In recent years there has been increasing success at applying the ideas of
causal mediation analysis to identify causally important nodes in deep neural networks, in particular
via the method of activation patching, where the output of a model component is intervened on.
This technique has been widely used by the community and successfully applied in a range of
contexts (Conmy et al., 2023; Cunningham et al., 2023; Feng and Steinhardt, 2023; Finlayson et al.,
2021; Geva et al., 2023; Goldowsky-Dill et al., 2023; Hanna et al., 2023; Hase et al., 2023; Hendel
et al., 2023; Huang et al., 2023; Lieberum et al., 2023; McDougall et al., 2023; Meng et al., 2023;
Merullo et al., 2023; Nanda et al., 2023; Olsson et al., 2022; Soulos et al., 2020; Stolfo et al., 2023;
Tigges et al., 2023; Todd et al., 2023; Vig et al., 2020; Wang et al., 2022).
Chan et al. (2022) introduce causal scrubbing, a generalized algorithm to verify a hypothesis
about the internal mechanism underlying a model’s behavior, and detail their motivation behind
performing noising and resample ablation rather than denoising or using mean or zero ablation – they
interpret the hypothesis as implying the computation is invariant to some large set of perturbations,
so their starting-point is the clean unperturbed forward pass.13
Anotherlineofresearchconcerningformalizingcausalabstractionsfocusesonfindingandverifying
high-level causal abstractions of low-level variables (Geiger et al., 2020, 2021, 2022, 2023). See
Jenner et al. (2022) for more details on how these different frameworks agree and differ. In contrast
to those works, we are chiefly concerned with identifying the important low-level variables in the
computational graph and are not investigating their semantics or potential groupings of lower-level
into higher-level variables.
In addition to causal mediation analysis, intervening on node activations in the model forward
pass has also been studied as a way of steering models towards desirable behavior (Belrose et al.,
2023; Jorgensen et al., 2023; Li et al., 2023; Rimsky et al., 2023; Turner et al., 2023; Zou et al.,
2023).
Attribution Patching / Gradient-based MaskingWhile we use the resample–ablation variant of
AtP as formulated in Nanda (2022), similar formulations have been used in the past to successfully
prune deep neural networks (Figurnov et al., 2016; Michel et al., 2019; Molchanov et al., 2017),
or even identify causally important nodes for interpretability (Cao et al., 2021). Concurrent work
by Syed et al. (2023) also demonstrates AtP can help with automatically finding causally important
circuits in a way that agrees with previous manual circuit identification work. In contrast to Syed
et al. (2023), we provide further analysis of AtP’s failure modes, give improvements in the form of
AtP∗, and evaluate both methods as well as several baselines on a suite of larger models against a
ground truth that is independent of human researchers’ judgement.
13Our motivation for focusing on noising rather than denoising was a closely related one – we were motivated by
automated circuit discovery, where gradually noising more and more of the model is the basic methodology for both of the
approaches discussed in Section 5.3.
21

<!-- page 22 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
7. Conclusion
Inthispaper,wehaveexploredtheuseofattributionpatchingfornodepatcheffectevaluation. Wehave
compared attribution patching with alternatives and augmentations, characterized its failure modes,
and presented reliability diagnostics. We have also discussed the implications of our contributions
for other settings in which patching can be of interest, such as circuit discovery, edge localization,
coarse-grained localization, and causal abstraction.
Our results show that AtP* can be a more reliable and scalable approach to node patch effect
evaluation than alternatives. However, it is important to be aware of the failure modes of attribution
patching, such as cancellation and saturation. We explored these in some detail, and provided
mitigations, as well as recommendations for diagnostics to ensure that the results are reliable.
We believe that our work makes an important contribution to the field of mechanistic interpretabil-
ity and will help to advance the development of more reliable and scalable methods for understanding
the behavior of deep neural networks.
8. Author Contributions
János Kramár was research lead, and Tom Lieberum was also a core contributor – both were highly
involved in most aspects of the project. Rohin Shah and Neel Nanda served as advisors and gave
feedback and guidance throughout.
References
N. Belrose, D. Schneider-Joseph, S. Ravfogel, R. Cotterell, E. Raff, and S. Biderman. Leace: Perfect
linear concept erasure in closed form.arXiv preprint arXiv:2306.03819, 2023.
S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit,
U. S. Prashanth, E. Raff, A. Skowron, L. Sutawika, and O. van der Wal. Pythia: A suite for
analyzing large language models across training and scaling. In A. Krause, E. Brunskill, K. Cho,
B.Engelhardt, S.Sabato, andJ.Scarlett, editors, InternationalConferenceonMachineLearning, ICML
2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 ofProceedings of Machine Learning
Research, pages 2397–2430. PMLR, 2023. URLhttps://proceedings.mlr.press/v202/
biderman23a.html.
T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison,
A. Askell, R. Lasenby, Y. Wu, S. Kravec, N. Schiefer, T. Maxwell, N. Joseph, Z. Hatfield-Dodds,
A. Tamkin, K. Nguyen, B. McLean, J. E. Burke, T. Hume, S. Carter, T. Henighan, and C. Olah.
Towards monosemanticity: Decomposing language models with dictionary learning.Transformer
Circuits Thread, 2023. https://transformer-circuits.pub/2023/monosemantic-features/index.html.
N. Cammarata, S. Carter, G. Goh, C. Olah, M. Petrov, L. Schubert, C. Voss, B. Egan, and S. K. Lim.
Thread: Circuits. Distill, 2020. doi: 10.23915/distill.00024. https://distill.pub/2020/circuits.
N. D. Cao, L. Schmid, D. Hupkes, and I. Titov. Sparse interventions in language models with
differentiable masking, 2021.
L. Chan, A. Garriga-Alonso, N. Goldwosky-Dill, R. Greenblatt, J. Nitishinskaya, A. Radhakrishnan,
B. Shlegeris, and N. Thomas. Causal scrubbing, a method for rigorously testing interpretabil-
ity hypotheses. AI Alignment Forum, 2022. https://www.alignmentforum.org/posts/
JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing .
22

<!-- page 23 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
A. Conmy, A. N. Mavor-Parker, A. Lynch, S. Heimersheim, and A. Garriga-Alonso. Towards automated
circuit discovery for mechanistic interpretability, 2023.
H. Cunningham, A. Ewart, L. Riggs, R. Huben, and L. Sharkey. Sparse autoencoders find highly
interpretable features in language models, 2023.
J. Feng and J. Steinhardt. How do language models bind entities in context?, 2023.
M. Figurnov, A. Ibraimova, D. P. Vetrov, and P. Kohli. Perforatedcnns: Acceleration through elim-
ination of redundant convolutions. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Gar-
nett, editors,Advances in Neural Information Processing Systems, volume 29. Curran Associates,
Inc., 2016. URL https://proceedings.neurips.cc/paper_files/paper/2016/file/
f0e52b27a7a5d6a1a87373dffa53dbe5-Paper.pdf.
M. Finlayson, A. Mueller, S. Gehrmann, S. Shieber, T. Linzen, and Y. Belinkov. Causal analysis
of syntactic agreement mechanisms in neural language models. In C. Zong, F. Xia, W. Li, and
R. Navigli, editors,Proceedings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1:
Long Papers), pages 1828–1843, Online, Aug. 2021. Association for Computational Linguistics. doi:
10.18653/v1/2021.acl-long.144. URL https://aclanthology.org/2021.acl-long.144.
L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima,
S. Presser, and C. Leahy. The Pile: An 800gb dataset of diverse text for language modeling.arXiv
preprint arXiv:2101.00027, 2020.
A. Geiger, K. Richardson, and C. Potts. Neural natural language inference models partially embed
theories of lexical entailment and negation, 2020.
A. Geiger, H. Lu, T. Icard, and C. Potts. Causal abstractions of neural networks, 2021.
A. Geiger, Z. Wu, H. Lu, J. Rozner, E. Kreiss, T. Icard, N. D. Goodman, and C. Potts. Inducing causal
structure for interpretable neural networks, 2022.
A. Geiger, C. Potts, and T. Icard. Causal abstraction for faithful model interpretation, 2023.
M. Geva, J. Bastings, K. Filippova, and A. Globerson. Dissecting recall of factual associations in
auto-regressive language models, 2023.
N. Goldowsky-Dill, C. MacLeod, L. Sato, and A. Arora. Localizing model behavior with path patching,
2023.
W. Gurnee, N. Nanda, M. Pauly, K. Harvey, D. Troitskii, and D. Bertsimas. Finding neurons in a
haystack: Case studies with sparse probing, 2023.
M. Hanna, O. Liu, and A. Variengien. How does gpt-2 compute greater-than?: Interpreting mathe-
matical abilities in a pre-trained language model, 2023.
P. Hase, M. Bansal, B. Kim, and A. Ghandeharioun. Does localization inform editing? surprising
differences in causality-based localization vs. knowledge editing in language models, 2023.
R. Hendel, M. Geva, and A. Globerson. In-context learning creates task vectors, 2023.
J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas,
L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den
Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, O. Vinyals, J. Rae, and
23

<!-- page 24 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
L. Sifre. An empirical analysis of compute-optimal large language model training. In
S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in
Neural Information Processing Systems, volume 35, pages 30016–30030. Curran Associates,
Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/
c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf.
J. Huang, A. Geiger, K. D’Oosterlinck, Z. Wu, and C. Potts. Rigorously assessing natural language
explanations of neurons, 2023.
E. Jenner, A. Garriga-Alonso, and E. Zverev. A comparison of causal scrub-
bing, causal abstractions, and related methods. AI Alignment Forum ,
2022. https://www.alignmentforum.org/posts/uLMWMeBG3ruoBRhMW/
a-comparison-of-causal-scrubbing-causal-abstractions-and .
O. Jorgensen, D. Cope, N. Schoots, and M. Shanahan. Improving activation steering in language
models with mean-centring, 2023.
K. Li, O. Patel, F. Viégas, H. Pfister, and M. Wattenberg. Inference-time intervention: Eliciting truthful
answers from a language model, 2023.
T. Lieberum, M. Rahtz, J. Kramár, N. Nanda, G. Irving, R. Shah, and V. Mikulik. Does circuit analysis
interpretability scale? evidence from multiple choice capabilities in chinchilla, 2023.
C. Louizos, M. Welling, and D. P. Kingma. Learning sparse neural networks through𝑙0 regularization,
2018.
C. McDougall, A. Conmy, C. Rushing, T. McGrath, and N. Nanda. Copy suppression: Comprehensively
understanding an attention head, 2023.
T. McGrath, M. Rahtz, J. Kramár, V. Mikulik, and S. Legg. The hydra effect: Emergent self-repair in
language model computations, 2023.
K. Meng, D. Bau, A. Andonian, and Y. Belinkov. Locating and editing factual associations in gpt, 2023.
J. Merullo, C. Eickhoff, and E. Pavlick. Circuit component reuse across tasks in transformer language
models, 2023.
P. Michel, O. Levy, and G. Neubig. Are sixteen heads really better than one? In
H. Wallach, H. Larochelle, A. Beygelzimer, F. d 'Alché-Buc, E. Fox, and R. Garnett, ed-
itors, Advances in Neural Information Processing Systems , volume 32. Curran Associates,
Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/file/
2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf.
P. Molchanov, S. Tyree, T. Karras, T. Aila, and J. Kautz. Pruning convolutional neural networks for
resource efficient inference. InInternational Conference on Learning Representations, 2017. URL
https://openreview.net/forum?id=SJGCiw5gl.
N. Nanda. Attribution patching: Activation patching at industrial scale. 2022. URLhttps://www.
neelnanda.io/mechanistic-interpretability/attribution-patching.
N. Nanda, S. Rajamanoharan, J. Kramár, and R. Shah. Fact finding: At-
tempting to reverse-engineer factual recall on the neuron level, Dec 2023.
URL https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/
fact-finding-attempting-to-reverse-engineer-factual-recall .
24

<!-- page 25 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
nostalgebraist. interpreting gpt: the logit lens. 2020. URLhttps://www.alignmentforum.org/
posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.
C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai,
A. Chen, T. Conerly, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, S. Johnston, A. Jones,
J.Kernion, L.Lovitt, K.Ndousse, D.Amodei, T.Brown, J.Clark, J.Kaplan, S.McCandlish, andC.Olah.
In-context learning and induction heads.Transformer Circuits Thread, 2022. https://transformer-
circuits.pub/2022/in-context-learning-and-induction-heads/index.html.
J. Pearl.Causality: Models, Reasoning and Inference. Cambridge University Press, 2000.
J. Pearl. Direct and indirect effects, 2001.
A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever. Improving language understanding by
generative pre-training, 2018.
N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, and A. M. Turner. Steering llama 2 via
contrastive activation addition, 2023.
J. M. Robins and S. Greenland. Identifiability and exchangeability for direct and indirect ef-
fects. Epidemiology, 3:143–155, 1992. URLhttps://api.semanticscholar.org/CorpusID:
10757981.
P. Soulos, R. T. McCoy, T. Linzen, and P. Smolensky. Discovering the compositional structure
of vector representations with role learning networks. In A. Alishahi, Y. Belinkov, G. Chru-
pała, D. Hupkes, Y. Pinter, and H. Sajjad, editors,Proceedings of the Third BlackboxNLP Work-
shop on Analyzing and Interpreting Neural Networks for NLP, pages 238–254, Online, Nov. 2020.
Association for Computational Linguistics. doi: 10.18653/v1/2020.blackboxnlp-1.23. URL
https://aclanthology.org/2020.blackboxnlp-1.23.
A. Stolfo, Y. Belinkov, and M. Sachan. A mechanistic interpretation of arithmetic reasoning in language
models using causal mediation analysis, 2023.
A. Syed, C. Rager, and A. Conmy. Attribution patching outperforms automated circuit discovery, 2023.
C. Tigges, O. J. Hollinsworth, A. Geiger, and N. Nanda. Linear representations of sentiment in large
language models, 2023.
E. Todd, M. L. Li, A. S. Sharma, A. Mueller, B. C. Wallace, and D. Bau. Function vectors in large
language models, 2023.
A. M. Turner, L. Thiergart, D. Udell, G. Leech, U. Mini, and M. MacDiarmid. Activation addition:
Steering language models without optimization, 2023.
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin.
Attention is all you need, 2017.
A. Veit, M. J. Wilber, and S. Belongie. Residual networks behave like ensembles of rela-
tively shallow networks. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett,
editors, Advances in Neural Information Processing Systems, volume 29. Curran Associates,
Inc., 2016. URL https://proceedings.neurips.cc/paper_files/paper/2016/file/
37bc2f75bf1bcfe8450a1a41c200364c-Paper.pdf.
25

<!-- page 26 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
J. Vig, S. Gehrmann, Y. Belinkov, S. Qian, D. Nevo, Y. Singer, and S. Shieber. Investigating gender
bias in language models using causal mediation analysis. In H. Larochelle, M. Ranzato, R. Hadsell,
M. Balcan, and H. Lin, editors,Advances in Neural Information Processing Systems, volume 33,
pages 12388–12401. Curran Associates, Inc., 2020. URLhttps://proceedings.neurips.cc/
paper_files/paper/2020/file/92650b2e92217715fe312e6fa7b90d82-Paper.pdf.
K. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt. Interpretability in the wild: a
circuit for indirect object identification in gpt-2 small, 2022.
B. L. Welch. The generalization of ‘Student’s’ problem when several different population variances are
involved. Biometrika, 34(1-2):28–35, 01 1947. ISSN 0006-3444. doi: 10.1093/biomet/34.1-2.28.
URL https://doi.org/10.1093/biomet/34.1-2.28.
A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A.-K. Dombrowski,
S. Goel, N. Li, M. J. Byun, Z. Wang, A. Mallen, S. Basart, S. Koyejo, D. Song, M. Fredrikson, J. Z.
Kolter, and D. Hendrycks. Representation engineering: A top-down approach to ai transparency,
2023.
26

<!-- page 27 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
A. Method details
A.1. Baselines
A.1.1. Properties of Subsampling
Here we prove that the subsampling estimatorˆISS (𝑛) from Section 3.3 is unbiased in the case of no
interaction effects. Furthermore, assuming a simple interaction model, we show the bias ofˆISS (𝑛) is
𝑝 times the total interaction effect of𝑛 with other nodes. We assume a pairwise interaction model.
That is, given a set of nodes𝜂, we have
I (𝜂; 𝑥) =
∑︁
𝑛∈𝜂
I (𝑛; 𝑥) +
∑︁
𝑛,𝑛′ ∈𝜂
𝑛≠𝑛
𝜎𝑛,𝑛′ (𝑥) (14)
with fixed constants𝜎𝑛,𝑛′ (𝑥) ∈ ℝ for each prompt pair𝑥 ∈ support(D). Let 𝜎𝑛,𝑛′ = 𝔼𝑥∼D

𝜎𝑛,𝑛′ (𝑥)

.
Let 𝑝 be the probability of including each node in a given𝜂 and let 𝑀 be the number of node
masks sampled fromBernoulli| 𝑁 | ( 𝑝) and prompt pairs𝑥 sampled fromD. Then,
𝔼
h
ˆISS (𝑛)
i
= 𝔼
"
1
|𝜂+(𝑛)|
|𝜂+ (𝑛) |∑︁
𝑘=1
I (𝜂+
𝑘 (𝑛); 𝑥+
𝑘 ) − 1
|𝜂− (𝑛)|
|𝜂− (𝑛) |∑︁
𝑘=1
I (𝜂−
𝑘 (𝑛); 𝑥 −
𝑘 )
#
(15a)
= 𝔼

𝔼
"
1
|𝜂+(𝑛)|
|𝜂+ (𝑛) |∑︁
𝑘=1
I (𝜂+
𝑘 (𝑛); 𝑥+
𝑘 ) − 1
|𝜂− (𝑛)|
|𝜂− (𝑛) |∑︁
𝑘=1
I (𝜂−
𝑘 (𝑛); 𝑥 −
𝑘 )

|𝜂+(𝑛)|
#
(15b)
= 𝔼

𝔼
 |𝜂+(𝑛)|
|𝜂+(𝑛)| 𝔼 [I (𝜂1; 𝑥1)|𝑛 ∈ 𝜂1] − |𝜂− (𝑛)|
|𝜂− (𝑛)| 𝔼 [I (𝜂1; 𝑥1)|𝑛 ∉ 𝜂1]
|𝜂+(𝑛)|

(15c)
= 𝔼 [I (𝜂1; 𝑥1)|𝑛 ∈ 𝜂1] − 𝔼 [I (𝜂1; 𝑥1)|𝑛 ∉ 𝜂1] (15d)
= 𝑐(𝑛) + 𝔼

∑︁
𝑛′≠𝑛
𝟙[𝑛′ ∈ 𝜂1] ©­
«
𝑐(𝑛′) + 𝜎𝑛𝑛′ + 1
2
∑︁
𝑛′′∉{𝑛′,𝑛}
𝟙[𝑛′ ∈ 𝜂1]𝜎𝑛′𝑛′′

𝑛 ∈ 𝜂1
ª®
¬

(15e)
− 𝔼

∑︁
𝑛′≠𝑛
𝟙[𝑛′ ∈ 𝜂1] ©­
«
𝑐(𝑛′) + 1
2
∑︁
𝑛′′∉{𝑛′,𝑛}
𝟙[𝑛′ ∈ 𝜂1]𝜎𝑛′𝑛′′
ª®
¬

𝑛 ∉ 𝜂1

(15f)
= 𝑐(𝑛) + 𝑝
∑︁
𝑛′≠𝑛
𝜎𝑛𝑛′ (15g)
In Equation (15g), we observe that if the interaction terms𝜎𝑛𝑛′ are all zero, the estimator is
unbiased. Otherwise, the bias scales both with the sum of interaction effects and with𝑝, as expected.
A.1.2. Pseudocode for Blocks and Hierarchical baselines
In Algorithm 2 we detail the Blocks baseline algorithm. As explained in Section 3.3, it comes with a
tradeoff in its “block size” hyperparameter𝐵: a small block size requires a lot of time to evaluate all
the blocks, while a large block size means many irrelevant nodes to evaluate in each high-contribution
block.
The Hierarchical baseline algorithm aims to resolve this tradeoff, by using small blocks, but
grouped into superblocks so it’s not necessary to traverse all the small blocks before finding the key
27

<!-- page 28 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Algorithm 2Blocks algorithm for causal attribution.
Require: blocksize 𝐵, computebudget 𝑀, nodes𝑁 = {𝑛𝑖}, prompts𝑥clean, 𝑥 noise, interventionfunction
˜I : 𝜂 ↦→ I (𝜂; 𝑥clean, 𝑥noise)
1: numBlocks ← ⌈| 𝑁 |/𝐵⌉
2: 𝜋 ← shuffle ({⌊numBlocks · 𝑖𝐵/| 𝑁 |⌋ | 𝑖 ∈ {0, . . . , |𝑁 | − 1}}) ⊲ Assign each node to a block.
3: for 𝑖 ← 0 to numBlocks−1 do
4: blockContribution[𝑖] ← | ˜I (𝜋−1({ 𝑖}))| ⊲ 𝜋 −1({ 𝑖}) := {𝑛 : 𝜋(𝑛) = 𝑖 | 𝑛 ∈ 𝑁})
5: spentBudget ← 𝑀 −numBlocks
6: topNodeContribs ← CreateEmptyDictionary()
7: for all𝑖 ∈ {0 to numBlocks−1} in decreasing order of blockContribution[𝑖] do
8: for all𝑛 ∈ 𝜋−1({ 𝑖}) do ⊲ Eval all nodes in block.
9: if spentBudget < 𝑀 then
10: topNodeContribs[𝑛] ←| ˜I ({𝑛})|
11: spentBudget ← spentBudget +1
12: else
13: return topNodeContribs
14: return topNodeContribs
nodes. In Algorithm 3 we detail the hierarchical algorithm in its iterative form, corresponding to
batch size 1.
One aspect that might be surprising is that on line 21, we ensure a subblock is never added to the
priority queue with higher priority than its ancestor superblocks. The reason for doing this is that in
practice we use batched inference rather than patching a single block at a time, so depending on the
batch size, we do evaluate blocks that aren’t the highest-priority unevaluated blocks, and this might
impose a significant delay in when some blocks are evaluated. In order to reduce this dependence on
the batch size hyperparameter, line 21 ensures that every block is evaluated at most𝐿 batches later
than it would be with batch size 1.
A.2. AtP improvements
A.2.1. Pseudocode for corrected AtP on attention keys
As described in Section 3.1.1, computing Equation (10) naïvely for all nodes requiresO(𝑇3) flops at
each attention head and prompt pair. Here we give a more efficient algorithm running inO(𝑇2). In
addition to keys, queries and attention probabilities, we now also cache attention logits (pre-softmax
scaled key-query dot products).
We defineattnLogits𝑡
patch(𝑛𝑞) and Δ𝑡 attnLogits(𝑛𝑞) analogously to Equations (8) and (9). For
brevity we can also define attnLogitspatch(𝑛𝑞)𝑡 := attnLogits𝑡
patch(𝑛𝑞)𝑡 and ΔattnLogits(𝑛𝑞)𝑡 :=
Δ𝑡 attnLogits(𝑛𝑞)𝑡, since the aim with this algorithm is to avoid having to separately compute ef-
fects ofdo(𝑛𝑘
𝑡 ← 𝑛𝑘
𝑡 (𝑥noise)) on any other component ofattnLogits than the one for key node𝑛𝑘
𝑡.
Note that, for a key𝑛𝑘
𝑡 at position𝑡 in the sequence, the proportions of the non-𝑡 components of
attn(𝑛𝑞)𝑡 do not change whenattnLogits(𝑛𝑞)𝑡 is changed, soΔ𝑡 attn(𝑛𝑞) is actuallyonehot(𝑡) −attn(𝑛𝑞)
multiplied by some scalar𝑠𝑡; specifically, to get the right attention weight on𝑛𝑘
𝑡, the scalar must be
𝑠𝑡 := Δattn(𝑛𝑞 )𝑡
1−attn(𝑛𝑞 )𝑡
. Additionally, we havelog

attn𝑡
patch (𝑛𝑞 )𝑡
1−attn𝑡
patch (𝑛𝑞 )𝑡

= log

attn(𝑛𝑞 )𝑡
1−attn(𝑛𝑞 )𝑡

+ ΔattnLogits(𝑛𝑞)𝑡; note
that the logodds function𝑝 ↦→ log

𝑝
1− 𝑝

is the inverse of the sigmoid function, soattn𝑡
patch(𝑛𝑞) =
28

<!-- page 29 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Algorithm 3 Hierarchical algorithm for causal attribution, in iterative form. In practice we do
additional batching rather than evaluating a single block at a time on line 14.
Require: branching factor 𝐵, num levels𝐿, compute budget𝑀, nodes 𝑁 = {𝑛𝑖}, intervention function
I
1: numTopLevelBlocks ← ⌈| 𝑁 |/𝐵𝐿⌉
2: 𝜋 ← shuffle  
numTopLevelBlocks · 𝑖𝐵 𝐿/| 𝑁 |
𝑖 ∈ {0, . . . , |𝑁 | − 1}
    
3: for all𝑛𝑖 ∈ 𝑁 do
4: (𝑑𝐿−1, 𝑑𝐿−2, . . . , 𝑑0) ← zero-padded final 𝐿 base-𝐵 digits of𝜋𝑖
5: address(𝑛𝑖) = ( ⌊𝜋𝑖/𝐵𝐿⌋, 𝑑𝐿−1, . . . , 𝑑0)
6: 𝑄 ← CreateEmptyPriorityQueue ()
7: for 𝑖 ← 0 to numTopLevelBlocks−1 do
8: PriorityQueueInsert(𝑄, [𝑖], ∞)
9: spentBudget ← 0
10: topNodeContribs ← CreateEmptyDictionary()
11: repeat
12: (addressPrefix,priority) ← PriorityQueuePop(𝑄)
13: blockNodes ← {𝑛 ∈ 𝑁 |StartsWith(address(𝑛),addressPrefix)}
14: blockContribution ← |I (blockNodes) |
15: spentBudget ← spentBudget +1
16: if blockNodes = {𝑛} for some𝑛 ∈ 𝑁 then
17: topNodeContribs[𝑛] ← blockContribution
18: else
19: for 𝑖 ← 0 to 𝐵 −1 do
20: if {𝑛 ∈ blockNodes|StartsWith(address(𝑛),addressPrefix + [ 𝑖]} ≠ ∅then
21: PriorityQueueInsert(𝑄,addressPrefix + [ 𝑖],min(blockContribution,priority))
22: until spentBudget = 𝑀 or PriorityQueueEmpty (𝑄)
23: return topNodeContribs
29

<!-- page 30 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
𝜎

log

attn𝑡
patch (𝑛𝑞 )𝑡
1−attn𝑡
patch (𝑛𝑞 )𝑡

. Putting this together, we can compute allattnLogitspatch(𝑛𝑞) by combining
all keys from the𝑥noise forward pass with all queries from the𝑥clean forward pass, and proceed to
compute ΔattnLogits(𝑛𝑞), and allΔ𝑡 attn(𝑛𝑞)𝑡, and thus allˆI 𝐾
AtPfix(𝑛𝑡; 𝑥clean, 𝑥noise), usingO(𝑇2) flops
per attention head.
Algorithm 4 computes the contribution of some query node𝑛𝑞 and prompt pair𝑥clean, 𝑥noise to the
corrected AtP estimatesˆ𝑐𝐾
AtPfix(𝑛𝑘
𝑡 ) for key nodes𝑛𝑘
1, . . . , 𝑛𝑘
𝑇 from a single attention head, using𝑂(𝑇)
flops, while avoiding numerical overflows. We reuse the notationattn(𝑛𝑞), attn𝑡
patch(𝑛𝑞), Δ𝑡 attn(𝑛𝑞),
attnLogits(𝑛𝑞), attnLogitspatch(𝑛𝑞), and 𝑠𝑡 from Section 3.1.1, leaving the prompt pair implicit.
Algorithm 4AtP correction for attention keys
Require: a := attnLogits(𝑛𝑞), apatch := attnLogitspatch(𝑛𝑞), g := 𝜕L ( M (𝑥clean ) )
𝜕attn(𝑛𝑞 )
1: 𝑡∗ ← argmax𝑡 (𝑎𝑡)
2: ℓ ← a − 𝑎𝑡∗ −log (Í
𝑡 𝑒𝑎𝑡 −𝑎𝑡∗ ) ⊲ Clean log attn weights,ℓ = log(attn(𝑛𝑞))
3: d ← ℓ −log(1 − 𝑒ℓ) ⊲ Clean logodds,𝑑𝑡 = log

attn(𝑛𝑞 )𝑡
1−attn(𝑛𝑞 )𝑡

4: 𝑑𝑡∗ ← 𝑎𝑡∗ −max𝑡≠𝑡∗ 𝑎𝑡 −log  Í
𝑡′≠𝑡∗ 𝑒𝑎𝑡′ −max𝑡≠𝑡∗ 𝑎𝑡

⊲ Adjust d; more stable for𝑎𝑡∗ ≫ max𝑡≠𝑡∗ 𝑎𝑡
5: ℓpatch ← logsigmoid(d + apatch − a) ⊲ Patched log attn weights,ℓpatch
𝑡 = log(attn𝑡
patch(𝑛𝑞)𝑡)
6: Δℓ ← ℓpatch − ℓ ⊲ Δℓ𝑡 = log

attn𝑡
patch (𝑛𝑞 )𝑡
attn(𝑛𝑞 )𝑡

7: 𝑏 ← softmax(a)⊺g ⊲ 𝑏 = attn(𝑛𝑞)⊺g
8: for 𝑡 ← 1 to 𝑇 do
⊲ Compute scaling factor𝑠𝑡 := Δ𝑡 attn(𝑛𝑞 )𝑡
1−attn(𝑛𝑞 )𝑡
9: if ℓpatch
𝑡 > ℓ 𝑡 then ⊲ Avoid overflow whenℓpatch
𝑡 ≫ ℓ𝑡
10: 𝑠𝑡 ← 𝑒𝑑𝑡+Δℓ𝑡+log(1−𝑒−Δℓ𝑡 ) ⊲ 𝑠 𝑡 = attn(𝑛𝑞 )𝑡
1−attn(𝑛𝑞 )𝑡
attn𝑡
patch (𝑛𝑞 )𝑡
attn(𝑛𝑞 )𝑡

1 − attn(𝑛𝑞 )𝑡
attn𝑡
patch (𝑛𝑞 )𝑡

11: else ⊲ Avoid overflow whenℓpatch
𝑡 ≪ ℓ𝑡
12: 𝑠𝑡 ← −𝑒𝑑𝑡+log(1−𝑒Δℓ𝑡 ) ⊲ 𝑠 𝑡 = − attn(𝑛𝑞 )𝑡
1−attn(𝑛𝑞 )𝑡

1 −
attn𝑡
patch (𝑛𝑞 )𝑡
attn(𝑛𝑞 )𝑡

13: 𝑟𝑡 ← 𝑠𝑡 (𝑔𝑡 − 𝑏) ⊲ 𝑟 𝑡 = 𝑠𝑡 (onehot(𝑡) − attn(𝑛𝑞)) ⊺g = Δ𝑡 attn(𝑛𝑞) · 𝜕L ( M (𝑥clean ) )
𝜕attn(𝑛𝑞 )
14: return r
The corrected AtP estimatesˆ𝑐𝐾
AtPfix(𝑛𝑘
𝑡 ) can then be computed using Equation (10); in other words,
by summing the returned𝑟𝑡 from Algorithm 4 over queries𝑛𝑞 for this attention head, and averaging
over 𝑥clean, 𝑥noise ∼ D.
A.2.2. Properties of GradDrop
In Section 3.1.2 we introduced GradDrop to address an AtP failure mode arising from cancellation
between direct and indirect effects: roughly, if the total effect (on some prompt pair) isI (𝑛) =
Idirect(𝑛) + Iindirect(𝑛), and these are close to cancelling, then a small multiplicative approximation
error in ˆIindirect
AtP (𝑛), due to nonlinearities, can accidentally cause| ˆIdirect
AtP (𝑛) + ˆIindirect
AtP (𝑛)| to be orders
of magnitude smaller than|I (𝑛)|.
To address this failure mode with an improved estimatorˆ𝑐AtP+GD(𝑛), there’s 3 desiderata for
GradDrop:
1. ˆ𝑐AtP+GD(𝑛) shouldn’t be much smaller thanˆ𝑐AtP(𝑛), because that would risk creating more false
30

<!-- page 31 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
negatives.
2. ˆ𝑐AtP+GD(𝑛) should usually not be much larger thanˆ𝑐AtP(𝑛), because that would create false
positives, which also slows down verification and can effectively create false negatives at a given
budget.
3. If ˆ𝑐AtP(𝑛) is suffering from the cancellation failure mode, thenˆ𝑐AtP+GD(𝑛) should be significantly
larger thanˆ𝑐AtP(𝑛).
Let’s recall how GradDrop was defined in Section 3.1.2, using a virtual node𝑛out
ℓ to represent the
residual-stream contributions of layerℓ:
ˆ𝑐AtP+GD(𝑛) := 𝔼𝑥clean,𝑥noise
"
1
𝐿 −1
𝐿∑︁
ℓ=1
ˆIAtP+GDℓ (𝑛; 𝑥clean, 𝑥noise)

#
= 𝔼𝑥clean,𝑥noise
"
1
𝐿 −1
𝐿∑︁
ℓ=1
(𝑛(𝑥noise) − 𝑛(𝑥clean)) ⊺ 𝜕Lℓ
𝜕𝑛

#
= 𝔼𝑥clean,𝑥noise
"
1
𝐿 −1
𝐿∑︁
ℓ=1
(𝑛(𝑥noise) − 𝑛(𝑥clean)) ⊺ 𝜕L
𝜕𝑛 (M (𝑥clean | do(𝑛out
ℓ ← 𝑛out
ℓ (𝑥clean))))

#
To better understand the behaviour of GradDrop, let’s look more carefully at the gradient𝜕L
𝜕𝑛 .
The total gradient𝜕L
𝜕𝑛 can be expressed as a sum of all path gradients from the node𝑛 to the output.
Each path is characterized by the set of layers𝑠 it goes through (in contrast to routing via the skip
connection). We write the gradient along a path𝑠 as 𝜕L𝑠
𝜕𝑛 .
Let S be the set of all subsets of layers after the layer𝑛 is in. For example, the direct-effect path is
given by∅ ∈ S. Then the total gradient can be expressed as
𝜕L
𝜕𝑛 =
∑︁
𝑠∈ S
𝜕L𝑠
𝜕𝑛 . (16)
We can analogously define ˆI 𝑠
AtP(𝑛) = (𝑛(𝑥noise) − 𝑛(𝑥clean)) ⊺ 𝜕L𝑠
𝜕𝑛 , and break down ˆIAtP(𝑛) =Í
𝑠∈ S ˆI 𝑠
AtP(𝑛). The effect of doing GradDrop at some layerℓ is then to drop all termsˆI 𝑠
AtP(𝑛) with ℓ ∈ 𝑠:
in other words,
ˆIAtP+GDℓ (𝑛) =
∑︁
𝑠∈ S
ℓ∉𝑠
ˆI 𝑠
AtP(𝑛). (17)
Now we’ll use this understanding to discuss the 3 desiderata.
Firstly, most node effects are approximately independent of most layers (see e.g. Veit et al.
(2016)); for any layerℓ that 𝑛’s effect is independent of, we’ll haveˆIAtP+GDℓ (𝑛) = ˆIAtP(𝑛). Letting
𝐾 be the set of downstream layers that matter, this guarantees1
𝐿−1
Í𝐿
ℓ=1
ˆIAtP+GDℓ (𝑛; 𝑥clean, 𝑥noise)
 ≥
𝐿− |𝐾 | −1
𝐿−1
ˆIAtP(𝑛; 𝑥clean, 𝑥noise)
, which meets the first desideratum.
Regarding the second desideratum: for eachℓ we have
ˆIAtP+GDℓ (𝑛)
 ≤ Í
𝑠∈ S
ˆI 𝑠
AtP(𝑛)
, so overall
we have 1
𝐿−1
Í𝐿
ℓ=1
ˆIAtP+GDℓ (𝑛)
 ≤ 𝐿− |𝐾 | −1
𝐿−1
ˆIAtP(𝑛)
 + | 𝐾 |
𝐿−1
Í
𝑠∈ S
ˆI 𝑠
AtP(𝑛)
. For the RHS to be much larger
31

<!-- page 32 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
(e.g. 𝛼 times larger) than
Í
𝑠∈ S ˆI 𝑠
AtP(𝑛)
 = | ˆIAtP(𝑛)|, there must be quite a lot of cancellation between
different paths, enough so thatÍ
𝑠∈ S
ˆI 𝑠
AtP(𝑛)
 ≥ (𝐿−1)𝛼
| 𝐾 |
Í
𝑠∈ S ˆI 𝑠
AtP(𝑛)
. This is possible, but seems
generally unlikely for e.g.𝛼 > 3.
Now let’s consider the third desideratum, i.e. suppose𝑛 is a cancellation false negative, with
| ˆIAtP(𝑛)| ≪ |I ( 𝑛)| ≪ |I direct(𝑛)| ≈ | ˆIdirect
AtP (𝑛)|. Then,
Í
𝑠∈ S\∅ ˆI 𝑠
AtP(𝑛)
 =
ˆIAtP(𝑛) − ˆIdirect
AtP (𝑛)
 ≫ |I ( 𝑛)|.
Thesummandsin Í
𝑠∈ S\∅ ˆI 𝑠
AtP(𝑛) aretheunionofthesummandsin Í
𝑠∈ S
ℓ∈𝑠
ˆI 𝑠
AtP(𝑛) = ˆIAtP(𝑛)− ˆIAtP+GDℓ (𝑛)
across layers ℓ.
It’s then possible but intuitively unlikely thatÍ
ℓ
ˆIAtP(𝑛) − ˆIAtP+GDℓ (𝑛)
 would be much smaller
than
ˆIAtP(𝑛) − ˆIdirect
AtP (𝑛)
. Suppose the ratio is 𝛼, i.e. suppose Í
ℓ
ˆIAtP(𝑛) − ˆIAtP+GDℓ (𝑛)
 =
𝛼
ˆIAtP(𝑛) − ˆIdirect
AtP (𝑛)
. For example, if all indirect effects use paths of length 1 then the union is
a disjoint union, soÍ
ℓ
ˆIAtP(𝑛) − ˆIAtP+GDℓ (𝑛)
 ≥
Í
ℓ

ˆIAtP(𝑛) − ˆIAtP+GDℓ (𝑛)
 =
ˆIAtP(𝑛) − ˆIdirect
AtP (𝑛)
, so
𝛼 ≥ 1. Now:
∑︁
ℓ∈ 𝐾
ˆIAtP+GDℓ (𝑛)
 ≥
∑︁
ℓ∈ 𝐾
ˆIAtP(𝑛) − ˆIAtP+GDℓ (𝑛)
 − | 𝐾|
ˆIAtP(𝑛)
 (18)
= 𝛼
ˆIAtP(𝑛) − ˆIdirect
AtP (𝑛)
 − | 𝐾|
ˆIAtP(𝑛)
 (19)
≥ 𝛼
ˆIdirect
AtP (𝑛)
 − (| 𝐾| + 𝛼)
ˆIAtP(𝑛)
 (20)
∴ 1
𝐿 −1
𝐿∑︁
ℓ=1
ˆIAtP+GDℓ (𝑛)
 = 1
𝐿 −1
∑︁
ℓ∈ 𝐾
ˆIAtP+GDℓ (𝑛)
 + 𝐿 − | 𝐾| − 1
𝐿 −1
ˆIAtP(𝑛)
 (21)
≥ 𝛼
𝐿 −1
ˆIdirect
AtP (𝑛)
 + 𝐿 −2|𝐾| − 1 − 𝛼
𝐿 −1
ˆIAtP(𝑛)
 (22)
(23)
And the RHS is an improvement over
ˆIAtP(𝑛)
 so long as𝛼
ˆIdirect
AtP (𝑛)
 > (2|𝐾| + 𝛼)
ˆIAtP(𝑛)
, which
is likely given the assumptions.
Ultimately, though, the desiderata are validated by the experiments, which consistently show
GradDrops either decreasing or leaving untouched the number of false negatives, and thus improving
performance apart from the initial upfront cost of the extra backwards passes.
A.3. Algorithm for computing diagnostics
Given summary statistics¯𝑖±, 𝑠± and count± for every node 𝑛, obtained from Algorithm 1, and a
threshold 𝜃 > 0 we can use Welch’s𝑡-test Welch (1947) to test the hypothesis that|¯𝑖+ − ¯𝑖− | ≥ 𝜃.
Concretely we compute the𝑡-statistic via
𝑠¯𝑖± = 𝑠±√count±
(24)
𝑡 = 𝜃 − |¯𝑖+ −¯𝑖− |√︃
𝑠2
¯𝑖+
+ 𝑠2
¯𝑖−
. (25)
32

<!-- page 33 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
The effective degrees of freedom𝜈 can be approximated with the Welch–Satterthwaite equation
𝜈Welch =

𝑠2
+
count+
+ 𝑠2
−
count−
2
𝑠4
+
count2+ (count+ −1) + 𝑠4−
count2− (count− −1)
(26)
We then compute the probability (𝑝-value) of obtaining a𝑡 at least as large as observed, using the
cumulative distribution function of Student’s𝑡

𝑥; 𝜈Welch

at the appropriate points. We take the max
of the individual𝑝-values of all nodes to obtain an aggregate upper bound. Finally, we use binary
search to find the largest threshold𝜃 that still has an aggregate𝑝-value smaller than a given target
𝑝 value. We show multiple such diagnostic curves in Appendix B.3, for different confidence levels
(1 − 𝑝target).
B. Experiments
B.1. Prompt Distributions
B.1.1. IOI
We use the following prompt template:
BOS When ␣[A]␣and␣[B]␣went␣to␣the␣bar, ␣[A/C]␣gave␣a␣drink␣to ␣[B/A]
Each clean prompt𝑥clean uses two names A and B with completion B, while a noise prompt𝑥noise
uses names A, B, and C with completion A. We construct all possible such assignments where names
are chosen from the set of {Michael, Jessica, Ashley, Joshua, David, Sarah}, resulting in 120
prompt pairs.
B.1.2. A-AN
We use the following prompt template to induce the prediction of an indefinite article.
BOS I ␣want␣one␣pear. ␣Can␣you␣pick␣up␣a␣pear␣for␣me?
␣I␣want␣one␣orange. ␣Can␣you␣pick␣up␣an␣orange␣for␣me?
␣I␣want␣one␣[OBJECT]. ␣Can␣you␣pick␣up ␣[a/an]
Wefoundthatzeroshotperformanceofsmallmodelswasrelativelylow, butperformanceimproved
drastically when providing a single example of each case. Model performance was sensitive to the
ordering of the two examples but was better than random in all cases. The magnitude and sign of the
impact of the few-shot ordering was inconsistent.
Clean prompts 𝑥clean contain objects inducing ‘␣a’, one of {boat, coat, drum, horn, map, pipe,
screw, stamp, tent, wall}. Noise prompts 𝑥noise contain objects inducing ‘␣an’, one of {apple,
ant, axe, award, elephant, egg, orange, oven, onion, umbrella}. This results in a total of 100
prompt pairs.
B.2. Cancellation across a distribution
As mention in Section 2, we average the magnitudes of effects across a distribution, rather than
taking the magnitude of the average effect. We do this because cancellation of effects is happening
33

<!-- page 34 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
(a) Pythia-410M
 (b) Pythia-1B
(c) Pythia-2.8B
 (d) Pythia-12B
Figure 10 | Cancellation ratio acrossIOI for various model sizes. A ratio of 1 means positive and
negative effects cancel out across the distribution, whereas a ratio of 0 means only either negative or
positive effects exist across the distribution. We report cancellation ratio for different percentiles of
nodes based onÍ
𝑥clean,𝑥noise
I (𝑛; 𝑥clean, 𝑥noise)
.
frequently across a distribution, which, together with imprecise estimates, could lead to significant
false negatives. A proper ablation study to quantify this effect exactly is beyond the scope of this
work. In Figure 10, we show the degree of cancellation across the IOI distribution for various model
sizes. For this we define theCancellation Ratioof node𝑛 as
1 −
Í
𝑥clean,𝑥noise I (𝑛; 𝑥clean, 𝑥noise)

Í
𝑥clean,𝑥noise
I (𝑛; 𝑥clean, 𝑥noise)
 .
B.3. Additional detailed results
We show the diagnostic measurements for Pythia-12B across all investigated distributions in Figure 11,
and cost of verified 100% recall curves for all models and settings in Figures 12 and 13.
B.4. Metrics
In this paper we focus on the difference in loss (negative log probability) as the metricL. We provide
some evidence that AtP(∗) is not sensitive to the choice ofL. For Pythia-12B, onIOI-PP and IOI,
we show the rank scatter plots in Figure 14 for three different metrics.
For IOI, we also show that performance of AtP∗ looks notably worse when effects are evaluated
34

<!-- page 35 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure 11 | Diagnostic of false negatives for 12B across distributions.
(a.i) IOI-PP
 (a.ii) RAND-PP
 (a.iii) IOI
(a) AttentionNodes
(b.i) CITY-PP
 (b.ii) RAND-PP
 (b.iii) A-AN
(b) NeuronNodes
via denoising instead of noising (cf. Section 2.1). As of now we do not have a satisfactory explanation
for this observation.
B.5. Hyperparameter selection
The iterative baseline, and theAtP-based methods, have no hyperparameters. In general, we used 5
random seeds for each hyperparameter setting, and selected the setting that produced the lowest
IRWRGM cost (see Section 4.2).
For Subsampling, the two hyperparameters are the Bernoulli sampling probability𝑝, and the
number of samples to collect before verifying nodes in decreasing order ofˆ𝑐SS. 𝑝 was chosen from
{0.01, 0.03}14. The number of steps was chosen among power-of-2 numbers of batches, where the
batch size depended on the setting.
ForBlocks, we swept across block sizes 2, 6, 20, 60, 250. ForHierarchical, we used a branching
factor of 𝐵 = 3, because of the following heuristic argument. If all but one node had zero effect, then
discovering that node would be a matter of iterating through the hierarchy levels. We’d have number
of levelslog𝐵 |𝑁 |, and at each level,𝐵 forward passes would be required to find which lower-level
block the special node is in – and thus the cost of finding the node would be𝐵log𝐵 |𝑁 | = 𝐵
log 𝐵 log |𝑁 |.
𝐵
log 𝐵 is minimized at 𝐵 = 𝑒, or at 𝐵 = 3 if 𝐵 must be an integer. The other hyperparameter is the
number of levels; we swept this from 2 to 12.
14We observed early on that larger values of𝑝were consistently underperforming. We leave it to future work to investigate
more granular and smaller values for𝑝.
35

<!-- page 36 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure 12| Cost of verified 100% recall curves, sweeping across models and settings forNeuronNodes
(a.i) Pythia 410M
 (a.ii) Pythia 1B
 (a.iii) Pythia 2.8B
 (a.iv) Pythia 12B
(a) CITY-PP
(b.i) Pythia 410M
 (b.ii) Pythia 1B
 (b.iii) Pythia 2.8B
 (b.iv) Pythia 12B
(b) RAND-PP
(c.i) Pythia 410M
 (c.ii) Pythia 1B
 (c.iii) Pythia 2.8B
 (c.iv) Pythia 12B
(c) A-AN distribution
36

<!-- page 37 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure13 | Costofverified100%recallcurves, sweepingacrossmodelsandsettingsfor AttentionNodes
(a.i) Pythia 410M
 (a.ii) Pythia 1B
 (a.iii) Pythia 2.8B
 (a.iv) Pythia 12B
(a) IOI-PP
(b.i) Pythia 410M
 (b.ii) Pythia 1B
 (b.iii) Pythia 2.8B
 (b.iv) Pythia 12B
(b) RAND-PP
(c.i) Pythia 410M
 (c.ii) Pythia 1B
 (c.iii) Pythia 2.8B
 (c.iv) Pythia 12B
(c) IOI distribution
37

<!-- page 38 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure 14 | True ranks against AtP∗ ranks on Pythia-12B using various metricsL. The last row shows
the effect in the denoising (rather than noising) setting; we speculate that the lower-right subplot
(log-odds denoising) is similar to the lower-middle one (logit-diff denoising) because IOI produces a
bimodal distribution over the correct and alternate next token.
38

<!-- page 39 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
C. AtP variants
C.1. Residual-site AtP and Layer normalization
Let’s consider the behaviour of AtP on sites that contain much or all of the total signal in the residual
stream, such as residual-stream sites. Nanda (2022) described a concern about this behaviour: that
linear approximation of the layer normalization would do poorly if the patched value is significantly
different than the clean one, but with a similar norm. The proposed modification to AtP to account
for this was to hold the scaling factors (in the denominators) fixed when computing the backwards
pass. Here we’ll present an analysis of how this modification would affect the approximation error of
AtP. (Empirical investigation of this issue is beyond the scope of this paper.)
Concretely, let the node under consideration be𝑛, with clean and alternate values𝑛clean and
𝑛noise; and for simplicity, let’s assume the model does nothing more than an unparametrized RM-
SNorm M (𝑛) := 𝑛/|𝑛|. Let’s now consider how wellM (𝑛noise) is approximated, both by its first-
order approximation ˆMAtP(𝑛noise) := M (𝑛clean) + M ( 𝑛clean)⊥(𝑛noise − 𝑛clean) where M (𝑛clean)⊥ =
𝐼 − M (𝑛clean)M (𝑛clean)⊺ is the projection to the hyperplane orthogonal toM (𝑛clean), and by the
variant that fixes the denominator:ˆMAtP+frozenLN(𝑛noise) := 𝑛noise/|𝑛clean|.
To quantify the error in the above, we’ll measure the error𝜖 in terms of Euclidean distance. Let’s
also assume, without loss of generality, that|𝑛clean| = 1. Geometrically, then,M (𝑛) is a projection onto
the unit hypersphere,MAtP(𝑛) is a projection onto the tangent hyperplane at𝑛clean, andMAtP+frozenLN
is the identity function.
Now, let’sdefineorthogonalcoordinates(𝑥, 𝑦 ) ontheplanespannedby 𝑛clean, 𝑛noise, suchthat 𝑛clean
is mapped to(1,0) and 𝑛noise is mapped to(𝑥, 𝑦 ), with 𝑦 ≥ 0. Then, 𝜖AtP :=
 ˆM (𝑛noise) − M (𝑛noise)
 =
√︂
2 + 𝑦2 −2 𝑥+𝑦2
√
𝑥2+𝑦2
, while𝜖AtP+frozenLN :=
 ˆMfix(𝑛noise) − M (𝑛noise)
 =

√︁
𝑥2 + 𝑦2 −1
.
Plotting the error in Figure 15, we can see that, as might be expected, freezing the layer norm
denominators helps whenever𝑛noise indeed has the same norm as𝑛clean, and (barring weird cases with
𝑥 > 1) whenever the cosine-similarity is less than1
2; but largely hurts if𝑛noise is close to𝑛clean. This
illustrates that, while freezing the denominators will generally be unhelpful when patch distances
are small relative to the full residual signal (as with almost all nodes considered in this paper), it will
likely be helpful in a different setting of patching residual streams, which could be quite unaligned
but have similar norm.
C.2. Edge AtP and AtP*
Here we will investigate edge attribution patching, and how the cost scales if we use GradDrop and/or
QK fix. (For this section we’ll focus on a single prompt pair.)
First, let’s review what edge attribution patching is trying to approximate, and how it works.
C.2.1. Edge intervention effects
Given nodes𝑛1, 𝑛2 where 𝑛1 is upstream of𝑛2, if we were to patch in an alternate value for𝑛1, this
could impact 𝑛2 in a complicated nonlinear way. As discussed in 3.1.2, because LLMs have a residual
stream, the “direct effect” can be understood as the one holding all other possible intermediate nodes
between 𝑛1 and 𝑛2 fixed – and it’s a relatively simple function, composed of transforming the alternate
value 𝑛1(𝑥noise) to a residual stream contribution𝑟out,ℓ1 (𝑥clean|do(𝑛1 ← 𝑛1(𝑥noise))), then carrying it
along the residual stream to an input𝑟in,ℓ2 = 𝑟in,ℓ2 (𝑥clean) + ( 𝑟out,ℓ1 − 𝑟out,ℓ1 (𝑥clean)), and transforming
39

<!-- page 40 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
(a) 𝜖AtP
 (b) 𝜖AtP+frozenLN
(c) 𝜖AtP+frozenLN − 𝜖AtP
Figure 15 | A comparison of how AtP and AtP with frozen layernorm scaling behave in a toy setting
where the model we’re trying to approximate is justM (𝑛) := 𝑛/|𝑛|. The red region is where frozen
layernorm scaling helps; the blue region is where it hurts. We find that unless𝑥 > 1, frozen layernorm
scaling always has lower error when the cosine-similarity between𝑛noise and 𝑛clean is < 1
2 (in other
words the angle> 60◦), but often has higher error otherwise.
40

<!-- page 41 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
that into a value𝑛direct
2 .
In the above,ℓ1 and ℓ2 are the semilayers containing𝑛1 and 𝑛2, respectively. Let’s definen(ℓ1,ℓ2 )
to be the set of non-residual nodes between semilayersℓ1 and ℓ2. Then, we can define the resulting
𝑛direct
2 as:
𝑛directℓ1
2 (𝑥clean|do(𝑛1 ← 𝑛1(𝑥noise))) := 𝑛2(𝑥clean|do(𝑛1 ← 𝑛1(𝑥noise)) ,do(n(ℓ1,ℓ2 ) ← n(ℓ1,ℓ2 ) (𝑥clean))) .
The residual-stream input𝑟directℓ1
in,ℓ2
(𝑥clean|do(𝑛1 ← 𝑛1(𝑥noise))) is defined similarly.
Finally, 𝑛2 itself isn’t enough to compute the metricL – for that we also need to let the forward
pass M (𝑥clean) run using the modified𝑛directℓ1
2 (𝑥clean|do(𝑛1 ← 𝑛1(𝑥noise))), while removing all other
effects of𝑛1 (i.e. not patching it).
Writing this out, we have edge intervention effect
I (𝑛1 → 𝑛2; 𝑥clean, 𝑥noise) := L (M (𝑥clean|do(𝑛2 ← 𝑛directℓ1
2 (𝑥clean|do(𝑛1 ← 𝑛1(𝑥noise))))))
−L (M (𝑥clean)) . (27)
C.2.2. Nodes and Edges
Let’s briefly consider what edges we’d want to be evaluating this on. In Section 4.1, we were able
to conveniently separate attention nodes from MLP neurons, knowing that to handle both kinds of
nodes, we’d just need to be able handle each kind of node on its own, and then combine the results.
For edge interventions this of course isn’t true, because edges can go from MLP neurons to attention
nodes, and vice versa. For the purposes of this section, we’ll assume that the node set𝑁 contains the
attention nodes, and for MLPs either a node per layer (as in Syed et al. (2023)), or a node per neuron
(as in theNeuronNodes setting).
Regarding the edges, the MLP nodes can reasonably be connected with any upstream or down-
stream node, but this isn’t true for the attention nodes, which have more of a structure amongst
themselves: the key, query, and value nodes for an attention head can only affect downstream nodes
via the attention output nodes for that head, and vice versa. As a result, on edges between different
semilayers, upstream attention nodes must be attention head outputs, and downstream attention
nodes must be keys, queries, or values. In addition, there are some within-attention-head edges,
connecting each query node to the output node in the same position, and each key and value node to
output nodes in causally affectable positions.
C.2.3. Edge AtP
As with node activation patching, the edge intervention effectI (𝑛1 → 𝑛2; 𝑥clean, 𝑥noise) is costly to
evaluate directly for every edge, since a forward pass is required each time. However, as with AtP, we
can apply first-order approximations: we define
ˆIAtP(𝑛1 → 𝑛2; 𝑥clean, 𝑥noise) :=

Δ𝑟AtP
𝑛1 (𝑥clean, 𝑥noise)
 ⊺
∇AtP
𝑟𝑛2
L (M (𝑥clean)) , (28)
where Δ𝑟AtP
𝑛1 (𝑥clean, 𝑥noise) := Jac𝑛1 (𝑟out,ℓ1) (𝑛1(𝑥clean)) (𝑛1(𝑥noise) − 𝑛1(𝑥clean)) (29)
and ∇AtP
𝑟𝑛2
L (M (𝑥clean)) :=

Jac𝑟in,ℓ2
(𝑛2) (𝑟in,ℓ2 (𝑥clean))
 ⊺
∇𝑛2 (L (M (𝑥clean))) (𝑛2(𝑥clean)) , (30)
and this is a close approximation when𝑛1(𝑥noise) ≈ 𝑛1(𝑥clean).
41

<!-- page 42 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
A key benefit of this decomposition is that the first term depends only on𝑛1, and the second term
depends only on𝑛2; and they’re both easy to compute from a forward and backward pass on𝑥clean
and a forward pass on𝑥noise, just like AtP itself.
Then, to complete the edge-AtP evaluation, what remains computationally is to evaluate all
the dot products between nodes in different semilayers, at each token position. This requires
𝑑resid𝑇 (1−1
𝐿 )| 𝑁 |2/2multiplicationsintotal 15,where 𝐿isthenumberoflayers, 𝑇 isthenumberoftokens,
and |𝑁 | isthetotalnumberofnodes. Thiscostexceedsthecostofcomputingall Δ𝑟AtP
𝑛1 (𝑥clean, 𝑥noise) and
∇AtP
𝑟𝑛2
L (M (𝑥clean)) on Pythia 2.8B even with a single node per MLP layer; if we look at a larger model,
or especially if we consider single-neuron nodes even for small models, the gap grows significantly.
Due to this observation, we’ll focus our attention on the quadratic part of the compute cost,
pertaining to two nodes rather than just one – i.e. the number of multiplications in computing all
(Δ𝑟AtP
𝑛1 (𝑥clean, 𝑥noise)) ⊺∇AtP
𝑟𝑛2
L (M (𝑥clean)). Notably, we’ll also exclude within-attention-head edges from
the “quadratic cost”: these edges, from some key, query, or value node to an attention output node
can be handled by minor variations of the nodewise AtP or AtP* methods for the corresponding key,
query, or value node.
C.2.4. MLPs
There are a couple of issues that can come up around the MLP nodes. One is that, similarly to the
attention saturation issue described in Section 3.1.1, the linear approximation to the MLP may be
fairly bad in some cases, creating significant false negatives if𝑛2 is an MLP node. Another issue is
that if we use single-neuron nodes, then those are very numerous, making the𝑑resid-dimensional dot
product per edge quite costly.
MLP saturation and fix Just as clean activations that saturate the attention probability may have
small gradients that lead to strongly underestimated effects, the same is true of the MLP nonlinearity.
A similar fix is applicable: instead of using a linear approximation to the function from𝑛1 to 𝑛2, we
can linearly approximate the function from𝑛1 to the preactivation𝑛2,pre, and then recompute𝑛2
using that, before multiplying by the gradient.
This kind of rearrangement, where the gradient-delta-activation dot product is computed in𝑑𝑛2
dimensions rather than𝑑resid, will come up again – we’ll call it thefactored form of AtP.
Ifthenodesareneuronsthenthefactoredformrequiresnochangetothenumberofmultiplications;
however, if they’re MLP layers then there’s a large increase in cost, by a factor of𝑑neurons. This increase
is mitigated by two factors: one is that this is a small minority of edges, outnumbered by the number
of edges ending in attention nodes by3 × (# heads per layer); the other is the potential for parameter
sharing.
Neuron edges and parameter sharingA useful observation is that each edge, across different to-
ken16 positions,reusesthesameparametermatricesin Jac𝑛1 (𝑟out,ℓ1) (𝑛1(𝑥clean)) andJac𝑟in,ℓ2
(𝑛2) (𝑟in,ℓ2 (𝑥clean)).
Indeed, setting aside the MLP activation function, the only other nonlinearity in those functions is a
layer normalization; if we freeze the scaling factor at its clean value as in Appendix C.1, the Jacobians
are equal to the product of the corresponding parameter matrices, divided by the clean scaling factor.
Thus if we premultiply the parameter matrices then we eliminate the need to do so at each token,
15This formula omits edges within a single layer, for simplicity – but those are a small minority.
16Also across different batch entries, if we do this on more than one prompt pair.
42

<!-- page 43 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
which reduces the per-token quadratic cost by𝑑resid (i.e. to a scalar multiplication) for neuron-neuron
edges, or by𝑑resid/𝑑site (i.e. to a𝑑site-dimensional dot product) for edges between neurons and some
attention site.
It’s worth noting, though, that these premultiplied parameter matrices (or, indeed, the edge-AtP
estimates if we use neuron sites) will in total be many times (specifically,(𝐿 −1) 𝑑neurons
4𝑑resid
times) larger
than the MLP weights themselves, so storage may need to be considered carefully. It may be worth
considering ways to only find the largest estimates, or the estimates over some threshold, rather than
full estimates for all edges.
C.2.5. Edge AtP* costs
Let’s now consider how to adapt the AtP* proposals from Section 3.1 to this setting. We’ve already seen
that the MLP fix, which is similarly motivated to the QK fix, has negligible cost in the neuron-nodes
case, but comes with a𝑑neurons/𝑑resid overhead in quadratic cost in the case of using an MLP layer
per node, at least on edges into those MLP nodes. We’ll consider the MLP fix to be part of edge-AtP*.
Now let’s investigate the two corrections in regular AtP*: GradDrops, and the QK fix.
GradDrops GradDrops works by replacing the single backward pass in the AtP formula with𝐿
backward passes; this in effect means𝐿 values for the multiplicand∇AtP
𝑟𝑛2
L (M (𝑥clean)), so this is a
multiplicative factor of𝐿 on the quadratic cost (though in fact some of these will be duplicates, and
taking this into account lets us drive the multiplicative factor down to(𝐿 +1)/2). Notably this works
equally well with “factored AtP”, as used for neuron edges; and in particular, if𝑛2 is a neuron, the
gradients can easily be combined and shared across𝑛1s, eliminating the(𝐿 + 1)/2 quadratic-cost
overhead.
However, the motivation for GradDrops was to account for multiple paths whose effects may
cancel; in the edge-interventions setting, these can already be discovered in a different way (by
identifying the responsible edges out of𝑛2), so the benefit of GradDrops is lessened. At the same
time, the cost remains substantial. Thus, we’ll omit GradDrops from our recommended procedure
edge-AtP*.
QK fix The QK fix applies to the∇𝑛2 (L (M (𝑥clean))) (𝑛2(𝑥clean)) term, i.e. to replacing the linear
approximation to the softmax with a correct calculation to the change in softmax, for each different
input Δ𝑟AtP
𝑛1 (𝑥clean, 𝑥noise). As in Section 3.1.1, there’s the simpler case of accounting for𝑛2s that are
query nodes, and the more complicated case of𝑛2s that are key nodes using Algorithm 4 – but these
are both cheap to do after computing theΔattnLogits corresponding to𝑛2.
The “factored AtP” way is to matrix-multiplyΔ𝑟AtP
𝑛1 (𝑥clean, 𝑥noise) with key or query weights and
with the clean queries or keys, respectively. This means instead of the𝑑resid multiplications required
for each edge𝑛1 → 𝑛2 with AtP, we need𝑑resid𝑑key + 𝑇 𝑑key multiplications (which, thanks to the causal
mask, can be reduced to an average of𝑑key (𝑑resid + (𝑇 +1)/2)).
The “unfactored” option is to stay in the𝑟in,ℓ2 space: pre-multiply the clean queries or keys with
the respective key or query weight matrices, and then take the dot product ofΔ𝑟AtP
𝑛1 (𝑥clean, 𝑥noise) with
each one. This way, the quadratic part of the compute cost contains𝑑resid(𝑇 +1)/2 multiplications;
this will be more efficient for short sequence lengths.
This means that for edges into key and query nodes, the overhead of doing AtP+QKfix on the
quadratic cost is a multiplicative factor ofmin

𝑇+1
2 , 𝑑key

1 + 𝑇+1
2𝑑resid

.
43

<!-- page 44 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
AtP variant O→V O→Q,K O→MLP MLP→V MLP→Q,K MLP→MLP
MLP layers 𝐷𝐻2 2𝐷𝐻2 𝐷𝐻 𝐷𝐻 2𝐷𝐻 𝐷
QKfix 𝐷𝐻2 (𝑇 +1) 𝐷𝐻2 𝐷𝐻 𝐷𝐻 (𝑇 +1) 𝐷𝐻 𝐷
QKfix+GD 𝐿+1
2 𝐷𝐻2 (𝐿+1) (𝑇+1)
2 𝐷𝐻2 𝐿+1
2 𝐷𝐻 𝐿+1
2 𝐷𝐻 (𝐿+1) (𝑇+1)
2 𝐷𝐻 𝐿+1
2 𝐷
AtP* 𝐷𝐻2 (𝑇 +1) 𝐷𝐻2 𝑉 𝑁 𝐻 𝐷𝐻 (𝑇 +1) 𝐷𝐻 𝑁 𝐷
AtP*+GD 𝐿+1
2 𝐷𝐻2 (𝐿+1) (𝑇+1)
2 𝐷𝐻2 𝑉 𝑁 𝐻 𝐿+1
2 𝐷𝐻 (𝐿+1) (𝑇+1)
2 𝐷𝐻 𝑁 𝐷
QKfix (long) 𝐷𝐻2 (2𝐷 + 𝑇 +1) 𝐾 𝐻2 𝐷𝐻 𝐷𝐻 (2𝐷 + 𝑇 +1) 𝐾 𝐻 𝐷
QKfix+GD 𝐿+1
2 𝐷𝐻2 𝐿+1
2 (2𝐷 + 𝑇 +1) 𝐾 𝐻2 𝐿+1
2 𝐷𝐻 𝐿+1
2 𝐷𝐻 𝐿+1
2 (2𝐷 + 𝑇 +1) 𝐾 𝐻 𝐿+1
2 𝐷
ATP* 𝐷𝐻2 (2𝐷 + 𝑇 +1) 𝐾 𝐻2 𝑉 𝑁 𝐻 𝐷𝐻 (2𝐷 + 𝑇 +1) 𝐾 𝐻 𝑁 𝐷
AtP*+GD 𝐿+1
2 𝐷𝐻2 𝐿+1
2 (2𝐷 + 𝑇 +1) 𝐾 𝐻2 𝑉 𝑁 𝐻 𝐿+1
2 𝐷𝐻 𝐿+1
2 (2𝐷 + 𝑇 +1) 𝐾 𝐻 𝑁 𝐷
Neurons 𝐷𝐻2 2𝐷𝐻2 𝑉 𝑁 𝐻 𝑉 𝑁 𝐻 2𝐾 𝑁 𝐻 𝑁2
MLPfix 𝐷𝐻2 2𝐷𝐻2 𝑉 𝑁 𝐻 𝑉 𝑁 𝐻 2𝐾 𝑁 𝐻 𝑁2
AtP* 𝐷𝐻2 (𝑇 +1) 𝐷𝐻2 𝑉 𝑁 𝐻 𝑉 𝑁 𝐻 (𝑇 +1) 𝐾 𝑁 𝐻 𝑁2
AtP*+GD 𝐿+1
2 𝐷𝐻2 𝐿+1
2 (𝑇 +1) 𝐷𝐻2 𝑉 𝑁 𝐻 𝐿+1
2 𝑉 𝑁 𝐻 (𝐿+1) (𝑇+1)
2 𝐾 𝑁 𝐻 𝑁2
ATP* (long) 𝐷𝐻2 (2𝐷 + 𝑇 +1) 𝐾 𝐻2 𝑉 𝑁 𝐻 𝑉 𝑁 𝐻 (𝑇 +1) 𝐾 𝑁 𝐻 𝑁2
AtP*+GD 𝐿+1
2 𝐷𝐻2 𝐿+1
2 (2𝐷 + 𝑇 +1) 𝐾 𝐻2 𝑉 𝑁 𝐻 𝐿+1
2 𝑉 𝑁 𝐻 (𝐿+1) (𝑇+1)
2 𝐾 𝑁 𝐻 𝑁2
Table 2 | Per-token per-layer-pair total quadratic cost of each kind of between-layers edge, across edge-
AtP variants. For brevity, we omit the layer-pair  𝐿
2
 factor that would otherwise be in every cell, and use
𝐷 := 𝑑resid, 𝐻 := # heads per layer, 𝐾 := 𝑑key, 𝑉 := 𝑑value, 𝑁 := 𝑑neurons.
QK fix + GradDrops If the QK fix is being combined with GradDrops, then the first multiplication
by the 𝑑resid × 𝑑key matrix can be shared between the different gradients; so the overhead on the
quadratic cost of QKfix + GradDrops for edges into queries and keys, using the factored method, is
𝑑key

1 + (𝑇+1) (𝐿+1)
4𝑑resid

.
C.3. Conclusion
Considering all the above possibilities, it’s not obvious where the best tradeoff is between correct-
ness and compute cost in all situations. In Table 2 we provide formulas measuring the number of
multiplications in the quadratic cost for each kind of edge, across the variations we’ve mentioned. In
Figure 16 we plug in the 4 sizes of Pythia model used elsewhere in the paper, such as Figure 2, to
enable numerical comparison.
D. Distribution of true effects
In Figure 17, we show the distribution of𝑐(𝑛) across models and distributions.
44

<!-- page 45 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure 16 | A comparison of edge-AtP variants across model sizes and prompt lengths. AtP* here is defined
to include QKfix and MLPfix, but not GradDrops. The costs vary across several orders of magnitude for each
setting.
In the setting with full-MLP nodes, MLPfix carries substantial cost for short prompts, but barely matters for
long prompts.
In the neuron-nodes setting, MLPfix is costless. But GradDrops in that setting continues to impose a large cost;
even though it doesn’t affect MLP→MLP edges, it does affect MLP→Q,K edges, which come out dominating
the cost with QKfix.
45

<!-- page 46 -->

AtP∗: An efficient and scalable method for localizing LLM behaviour to components
Figure 17 | Distribution of true effects across models and prompt pair distributions
AttentionNodes NeuronNodes
(a.i)
 (a.ii)
(a) Pythia-410M
(b.i)
 (b.ii)
(b) Pythia-1B
(c.i)
 (c.ii)
(c) Pythia-2.8B
(d.i)
 (d.ii)
(d) Pythia-12B
46
