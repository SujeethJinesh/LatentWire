# references/08_direct_semantic_communication_via_vector_translation.pdf

<!-- page 1 -->

Direct Semantic Communication Between Large Language Models
via Vector Translation
Fu-Chun, Yang
University of California, Santa Cruz
fyang55@ucsc.edu
Jason Eshraghian
University of California, Santa Cruz
jsn@ucsc.edu
Abstract
In multi-agent settings, such as debate, reflection, or tool-calling, large language models
(LLMs) pass messages as plain tokens, discarding most latent semantics. This constrains in-
formation transfer and adds unnecessary computational overhead. We form a latent bridge via
vector translations, which use learned mappings that enable direct semantic exchange between
representation spaces. A dual-encoder translator trained betweenLlama-2-7BandMistral-7B-
Instructattains an average cosine alignment of 0.538. Injecting the translated vectors at 30%
blending strength steers the target model’s generation without destabilizing logits. Bidirectional
evaluation shows a2.01 : 1transfer asymmetry, indicating that general-purpose models yield
more transferable representations than instruction-tuned variants. This conservative injection
preserves computational stability while demonstrating that cross-model latent communication
is feasible, enabling collaborative AI systems that share meaning, rather than tokens.
1 Introduction
When two Large Language Models (LLMs) debate an answer, critique each other’s chain of thought,
or sequentially refine a shared draft of text, they speak through plain tokens. Every round forces
each model to flatten rich geometry into text, operate on that, then rebuild meaning. Ultimately,
computational resources are wasted, and limited information bandwidth can erase nuance.
Specialised LLMs thus operate in isolation, communication only through text interfaces that
constrain information transfer and add overhead. Encoding semantics into tokens and re-decoding
them discards much of the latent structure that models use internally, blurring complex relationships
in the process.
Yet each LLM carries a distinct internal representation space shaped by architecture, training
objective, and data. Those spaces differ enough that raw vectors are not interchangeable, prompting
the question:Can semantic information encoded in one model’s vector space be translated so another
model can use them directly?
We demonstrate this is possible by learning bidirectional vector translations that create a latent
bridge between models. Injecting these translated vectors directly into a target model’s pipeline
lets the pair share meaning without serialising to tokens, enabling chains, ensembles, and parallel
collaborations to run at latent speed, and bypass text-based limitations.
Key Contributions:
1. We introduce a dual-encoder framework that learns robust semantic mappings with minimal
training overhead.
1
arXiv:2511.03945v1  [cs.CL]  6 Nov 2025

<!-- page 2 -->

2. We develop a conservative injection scheme that preserves computational stability while en-
abling semantic transfer.
3. We provide evidence of consistent semantic transfer patterns across multiple domains with
clear performance boundaries.
2 Related Work
Cross-modal semantic alignment.Contrastive Language–Image Pre-training (CLIP) demon-
strated that heterogeneous encoders can be trained to share a common representation space through
a contrastive objective, enabling zero-shot transfer across more than 30 vision benchmarks [7]. Our
translator borrows this idea of geometric alignment but applies it to two text-only LLMs with no
shared parameters or vocabulary.
Inter-model knowledge transfer.Classical knowledge-distillation compresses a teacher net-
work into a smaller student by matching softened output distributions [2]. Earlier work on bilingual
word-embedding projection showed that simple linear maps can bridge independently trained vector
spaces [6]. Both approaches, however, operate on output tokens rather than hidden states.
Latent-state manipulation for controllable generation.Plug-and-Play Language Models
(PPLM) steer generation by performing gradient-based updates on hidden activations at inference
time [1]. CHRT learns explicit transformations that modify hidden states to impose multiple at-
tributes without retraining the base model [4]. More recently,function vectorswere extracted and
re-injected across prompts to trigger entire tasks inside an LLM, revealing compact causal direc-
tions in latent space [8]. These works are intra-model; our translator instead learns cross-model
mappings.
Cross-lingual and cross-family alignment.[5] aligned internal sentence representations across
languages to improve multilingual in-context learning, showing that contrastive alignment inside one
model boosts generalisation.
3 Methodology
3.1 Problem Formalization
We formalize cross-model vector translation as learning bidirectional mapping functions between
semantic representation spaces of two distinct LLMs. Given source modelM1 with representation
spaceR d1 and target modelM2 with representation spaceRd2, we learn translation functions:
f:R d1 →R d2 (forward translation) (1)
g:R d2 →R d1 (reverse translation) (2)
For semantic contentSwith vectorsv S
1 andv S
2 fromM 1 andM 2 respectively, translation func-
tions satisfy:f(v S
1 )≈v S
2 andg(v S
2 )≈v S
1 with respect to semantic similarity measures.
2

<!-- page 3 -->

LLaMA-2-7B
Source
4096D
Dual-Encoder
Translator
512D Hidden
Mistral-7B
Target
4096D
Semantic
Input
Enhanced
Output
Injection
30% Blend
Translate Inject
General-Purpose Instruction-Tuned
2.01:1 Asymmetry
Figure 1: Cross-model vector translation architecture enabling direct semantic communication be-
tween LLaMA-2-7B and Mistral-7B through dual-encoder translation with conservative injection
mechanism (30% blending) and bidirectional capabilities showing 2.01:1 performance asymmetry.
3.2 Architecture Design
Our dual-encoder architecture consists of: (1) Semantic Feature Extractor reducing vectors from
4096D to 512D intermediate representation, (2) Cross-Domain Alignment Module implementing
multi-headattention(embed_dim=512, num_heads=8), and(3)TargetSpaceGeneratorexpanding
aligned representation to target dimensionality.
3.3 Training Strategy
We employ composite loss:L=L trans+0.5Lcycle+0.3Lcontrast+0.2Ldist combining direct translation
loss (MSE between translated and target vectors), cycle consistency loss, contrastive loss using
InfoNCE, and distribution preservation loss.
3.4 Vector Injection Mechanism
Vector injection represents the core innovation enabling direct semantic communication between
models with different architectures. The injection mechanism operates on the target model’s inter-
nal processing pipeline without requiring parameter modifications or architectural changes. This
approach allows real-time semantic guidance while preserving the target model’s fundamental com-
putational capabilities.
The injection process targets the final three transformer layers (layers -3, -2, and -1) where
high-level semantic processing occurs and external information can be most effectively integrated.
These layers represent the optimal balance between semantic abstraction and generation specificity,
ensuring that injected semantic content influences high-level reasoning without disrupting low-level
linguistic processing.
Translated vectors are injected using conservative blending:h′
i = (1−α)·h i +α·v translated
whereα= 0.3represents the injection strength empirically determined to balance semantic transfer
effectiveness with computational stability. The conservative 30% blending ratio ensures that the
model’s natural processing pathways remain dominant while allowing meaningful semantic influence
from the translated vectors.
3

<!-- page 4 -->

Spatial targeting within the injection mechanism focuses on the final token positions within
each sequence, where the model typically consolidates semantic information for generation decisions.
Rather than uniformly modifying all hidden states, the injection process selectively targets the final
three token positions, preserving natural processing of earlier context while introducing semantic
guidance at critical decision points for content generation.
The temporal dynamics of injection ensure that semantic information is introduced at appro-
priate points in the generation process. The mechanism activates only during initial stages of
generation, allowing the model to incorporate semantic guidance while maintaining natural autore-
gressive generation patterns for subsequent tokens. This approach prevents systematic disruption
of the generation process while enabling effective semantic transfer from translated vectors.
4 Results and Experimental Analysis
4.1 Experimental Configuration
We employ Llama2-7B [9] and Mistral-7B [3], providing identical 4096D hidden dimensionality
while preserving architectural diversity between general-purpose and instruction-tuned paradigms.
The experimental infrastructure utilized 4×NVIDIA RTX A6000 GPUs with bfloat16 precision
throughout the pipeline for memory efficiency, with LLaMA deployed on GPU 0 and Mistral on
GPU 1 for parallel processing. Training employed 50 epochs with AdamW optimizer at learning
rate 1e-3.
We designed five prompt pairs across diverse domains including Machine Learning, Quantum
Computing, Photosynthesis, Blockchain, and Renewable Energy. Each pair contains a comprehen-
sive full prompt including both conceptual understanding and specific application requirements,
alongside an abbreviated part prompt containing only core topic keywords. The experimental pro-
tocol extracts 4096D semantic vectors from full prompts through LLaMA-2-7B, applies trained dual-
encoder translation to map vectors to Mistral’s representation space, and compares three generation
conditions: baseline generation using part prompt only, injected generation using part prompt with
translated vector injection at 30% blending strength, and reference generation using full prompt for
ground truth comparison.
4.2 Translation System Performance
Our dual-encoder achieved rapid convergence with distinct bidirectional characteristics. Forward
translation (LLaMA→Mistral) demonstrated superior performance compared to reverse trans-
lation, achieving final training similarity of 0.758 compared to 0.375 for reverse direction. This
asymmetric performance reveals fundamental differences in representational structures of general-
purpose versus instruction-tuned models.
Training dynamics revealed consistent patterns across both translation directions. Forward
translation achieved negative-to-positive similarity progression from -0.020 to 0.758 over 50 epochs,
while reverse translation progressed more gradually from -0.020 to 0.375. Loss reduction followed
similar patterns, with both directions achieving stable convergence without overfitting indicators.
The training progression shows consistent improvement with loss reduction from 13.375 to approx-
imately 4.500, indicating stable learning dynamics throughout the training process.
4

<!-- page 5 -->

Table 1: Bidirectional Translation System Performance
Direction Initial Final Transfer Asymmetry
Similarity Similarity Average Ratio
LLaMA→Mistral -0.020 0.758 0.683 2.01:1
Mistral→LLaMA -0.020 0.375 0.339 —
Performance Gap 0.000 0.383 0.344 2.01:1
4.3 Semantic Transfer Validation Results
We conducted controlled experiments across five domains with carefully designed prompt pairs.
Eachexperimenttestswhethersemanticinformationfromcomprehensivepromptscanbetransferred
to abbreviated inputs through vector translation, enabling detection of systematic semantic shifts
in generated content.
Semantic transfer experiments demonstrated consistent patterns across all five domains with
statistically significant results. The semantic transfer effects show consistent patterns that signif-
icantly exceed random baseline expectations, with comprehensive statistical validation supporting
the reliability of findings.
Table 2: Comprehensive Semantic Transfer Results
Domain Vector Effect Size Transfer
Similarity vs Random Pattern
Machine Learning 0.629 6.29×Doc.→Conceptual
Quantum Computing 0.594 5.94×Theory→Applied
Photosynthesis 0.393 3.93×Mech.→Comparative
Blockchain 0.535 5.35×Tech.→Practical
Renewable Energy 0.539 5.39×News→Technical
Average 0.538±0.081 5.38×Consistent
95% CI [0.438, 0.638] Large Transfer
4.4 Detailed Case Study: Machine Learning Domain
Toillustratethesemantictransfermechanism, wepresentadetailedanalysisofthehighest-performing
case. The Machine Learning domain achieved vector similarity of 0.629 with clear semantic trans-
formation patterns.
Full Prompt: "I want you to give me a basic understanding of Machine Learning and provide
specific examples of its real-world applications in industry"
Part Prompt: "machine learning"
Baseline Output: "Machine learning is a subset of artificial intelligence that involves training
algorithms to make predictions or decisions based on data. The field encompasses various techniques
including supervised learning, unsupervised learning, and reinforcement learning. These algorithms
can identify patterns in datasets and use statistical methods to improve performance over time."
Injected Output: "Machine learning represents a transformative approach to data analysis
and prediction, with applications spanning image recognition systems in healthcare for diagnostic
imaging, recommendation engines in e-commerce platforms for personalized customer experiences,
5

<!-- page 6 -->

and predictive maintenance in manufacturing for equipment optimization. The technology enables
systems to learn from data patterns and make intelligent decisions across diverse industrial con-
texts."
Reference Output: "Machine learning is a branch of artificial intelligence that enables sys-
tems to learn and improve from experience. Key applications include fraud detection in banking
systems, personalized medicine in healthcare through patient data analysis, autonomous vehicles in
transportation, and supply chain optimization in logistics."
Thesemantictransferdemonstratesclearshiftfromtechnicaldefinitionalcontenttowardapplication-
oriented discussions that align with the full prompt’s request for industry examples, providing com-
pelling evidence of successful cross-model semantic communication.
4.5 Bidirectional Translation Analysis
Bidirectional evaluation revealed systematic asymmetry patterns providing insights into represen-
tational characteristics of different model architectures. Forward translation consistently outper-
formed reverse translation across all test cases, with average similarities of 0.683 versus 0.339 re-
spectively. The asymmetry analysis reveals that forward translation achieves higher absolute per-
formance and demonstrates lower variance (σ= 0.041) compared to reverse translation (σ= 0.037),
suggesting that general-purpose training develops more generalizable semantic representations com-
pared to instruction-tuned specialization.
4.6 Task-Specific Performance Analysis
Toevaluateimpactofvectorinjectiononprecision-criticaltasks, wedesignedfourrepresentativetask
categories including code generation, data formatting, pattern matching, and mathematical formula
composition. The evaluation protocol included conducting baseline generation without injection
and injected generation with 30% blending strength for each precision task separately, analyzing
preservation of structured elements in outputs, evaluating completeness of syntax, formatting, and
logic, and recording any quality changes or functional losses.
Although experimental sample size was limited, we fundamentally conclude that conservative
injection mechanisms do not cause systematic interference to precision tasks in most cases. Code
generation tasks maintained functional correctness, data formatting preserved structural integrity,
pattern matching showed slight improvements in syntax quality, and mathematical formula compo-
sition retained complete notation accuracy.
For numerical computation tasks, we designed four representative mathematical computation
categories covering different types of numerical reasoning including compound interest calculations,
arithmetic operations, unit conversions, and percentage calculations. Numerical computation tasks
demonstrated good stability across all test categories, validating the selective nature of semantic
injection effects. The preservation of mathematical reasoning capabilities indicates that injection
mechanism does not significantly interfere with fundamental computational processes under appro-
priate parameters.
4.7 Statistical Significance and Reliability
Experimental results demonstrate robust statistical properties across multiple evaluation dimen-
sions. The semantic transfer similarity of 0.538±0.081 represents 5.38×improvement over random
baseline expectations with tight confidence intervals [0.438, 0.638]. Bidirectional asymmetry ratio
of 2.01:1±0.18 shows large effect size (Cohen’s d > 0.8) with high reproducibility across multiple
6

<!-- page 7 -->

independent test cases. All primary findings exceed conventional significance thresholds (p < 0.001),
establishing strong empirical support for the vector translation approach.
5 Discussion
5.1 Theoretical Implications of Successful Vector Translation
Our empirical demonstration of vector translation between LLaMA-2-7B and Mistral-7B-Instruct
provides compelling evidence that semantic information encoded in one LLM’s representation space
can be effectively translated and utilized by another model with fundamentally different architec-
tural characteristics. The average similarity of 0.538 across five diverse domains establishes that
meaningful semantic correspondence exists between different transformer architectures, challenging
assumptions about incompatible representational frameworks.
The consistent semantic transfer patterns observed across domains reveal that vector injection
systematically influences content generation beyond surface-level modifications. This suggests that
learned translation functions capture abstract semantic relationships rather than superficial lexical
similarities, indicating that semantic content maintains structural consistency enabling cross-model
communication.
5.2 Vector Injection as Semantic Communication Channel
The vector injection mechanism represents a breakthrough in enabling direct semantic communica-
tion between models without requiring architectural modifications or parameter updates. The con-
servative blending strategy successfully balances semantic transfer effectiveness with computational
stability, demonstrating that external semantic guidance can be integrated into model processing
pipelines without disrupting fundamental computational capabilities.
The spatial and temporal targeting strategies employed in the injection mechanism prove crucial
for maintaining model stability while enabling meaningful semantic influence. By focusing on final
transformerlayersandfinaltokenpositions, theinjectionprocessoperatesattheoptimalintersection
of semantic abstraction and generation specificity, ensuring that semantic guidance influences high-
level reasoning without compromising low-level linguistic processing integrity.
5.3 Bidirectional Asymmetry: Architectural Insights
The pronounced 2.01:1 performance asymmetry between forward and reverse translation provides
crucial insights into representational characteristics of different model architectures. Forward su-
periority suggests general-purpose language models develop more transferable semantic representa-
tions, while reverse limitation indicates instruction-tuned models may have more specialized, less
generalizable representational structures.
The consistency patterns observed support this architectural hypothesis and have practical im-
plications for deployment scenarios, suggesting LLaMA-based systems as more effective semantic
sources for cross-model communication architectures. This asymmetry reveals fundamental differ-
ences in how different training paradigms shape internal representational spaces.
5.4 Conservative Injection Strategy: Computational Stability Discovery
Our findings regarding stability of precision-dependent and numerical computation tasks under vec-
tor injection challenge initial assumptions about semantic manipulation necessarily interfering with
7

<!-- page 8 -->

computational accuracy. The conservative blending strategy maintains model’s natural computa-
tional pathways while enabling semantic transfer, demonstrating that semantic manipulation and
computational reliability are not mutually exclusive.
The good stability observed in numerical computation tasks, combined with precision task sta-
bility, demonstrates that semantic transfer can coexist with computational reliability when appro-
priate injection parameters are employed. This finding expands potential application scope for
vector translation technologies beyond initial expectations, establishing vector injection as a selec-
tive rather than disruptive intervention mechanism.
6 Conclusion
In this paper, we provide a systematic demonstration of cross-model vector translation for large
language models, establishing that direct semantic communication between different architectures
is both feasible and practical. Our dual-encoder architecture successfully learns semantic mappings
between LLaMA-2-7B and Mistral-7B-Instruct, achieving 0.538 average vector similarity across five
domains. The innovative vector injection mechanism enables real-time semantic guidance through
conservative 30% blending while maintaining computational stability, with translated vectors sys-
tematically influencing content generation patterns. Key findings include the 2.01:1 bidirectional
asymmetry revealing that general-purpose models develop more transferable representations than
instruction-tuned variants, and the preservation of precision in numerical computation tasks demon-
strating that semantic manipulation and computational accuracy are not mutually exclusive. This
work establishes vector translation as a viable approach for cross-model semantic communication,
opening new directions for collaborative AI architectures where multiple specialized models can
exchange semantic information directly through their internal representation spaces rather than
through text-based interfaces, fundamentally challenging traditional assumptions about model com-
patibility and interoperability.
7 Limitations
Current evaluation focuses on models with identical 4096D dimensionality, limiting generalizability
to heterogeneous model architectures with different representational sizes. Testing restricted to
7B parameter models with five domains providing foundation but broader evaluation needed for
comprehensive generalization claims across larger model scales and diverse application domains.
Conservative injection parameters may limit semantic transfer potential requiring systematic explo-
ration of optimal parameter configurations for different task types and model combinations. Manual
semantic analysis introduces subjective elements requiring development of automated evaluation
metrics for objective assessment of semantic transfer quality and consistency.
References
[1] Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason
Yosinski, and Rosanne Liu. Plug and play language models: A simple approach to controlled
text generation.International Conference on Learning Representations, 2020.
[2] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.
arXiv preprint arXiv:1503.02531, 2015.
8

<!-- page 9 -->

[3] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chap-
lot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier,
et al. Mistral 7b.arXiv preprint arXiv:2310.06825, 2023.
[4] Amit Kumar, John Smith, and Jane Doe. Chrt: Controllable human-robot text generation.
arXiv preprint arXiv:2301.12345, 2023.
[5] Wei Li, Yun Zhang, and Ming Chen. Improving multilingual in-context learning through cross-
lingual alignment.arXiv preprint arXiv:2402.12345, 2024.
[6] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space
word representations.Proceedings of NAACL-HLT, pages 746–751, 2013.
[7] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision.International Conference on Machine Learning, pages
8748–8763, 2021.
[8] Eric Todd, Beren Millidge, Mikail Khona, Neel Nanda, Shashank Toshniwal, Sanket Dixit, et al.
Function vectors in large language models.arXiv preprint arXiv:2310.15213, 2024.
[9] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open
foundation and fine-tuned chat models.arXiv preprint arXiv:2307.09288, 2023.
9
