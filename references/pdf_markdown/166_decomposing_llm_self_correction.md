# references/166_decomposing_llm_self_correction.pdf

<!-- page 1 -->

Decomposing LLM Self-Correction:
The Accuracy-Correction Paradox and Error Depth Hypothesis
Yin Li
University of Birmingham
kxl474@student.bham.ac.uk
January 6, 2026
Abstract
Large Language Models (LLMs) are widely believed to possess self-correction capabilities, yet re-
cent studies suggest thatintrinsicself-correction—where models correct their own outputs without ex-
ternal feedback—remains largely ineffective. In this work, we systematically decompose self-correction
into three distinct sub-capabilities:error detection,error localization, anderror correction. Through
cross-model experiments on GSM8K-Complex (n=500 per model, 346 total errors) with three major
LLMs, we uncover a strikingAccuracy-Correction Paradox:weakermodels (GPT-3.5, 66% accuracy)
achieve 1.6×higherintrinsic correction rates than stronger models (DeepSeek, 94% accuracy)—26.8%
vs 16.7%. We propose theError Depth Hypothesis: stronger models make fewer but “deeper” errors
that resist self-correction. Error detection rates vary dramatically across architectures (10% to 82%), yet
detection capability does not predict correction success—Claude detects only 10% of errors but corrects
29% intrinsically. Surprisingly, providing error location hintshurtsall models. Our findings challenge
linear assumptions about model capability and self-improvement, with important implications for the
design of self-refinement pipelines.
Keywords:Large Language Models, Self-Correction, Error Detection, Mathematical Reasoning, Model
Evaluation
1 Introduction
The ability to recognize and correct one’s own mistakes is a hallmark of intelligent reasoning. As Large Lan-
guage Models (LLMs) are increasingly deployed in high-stakes applications—from mathematical problem-
solving to code generation and scientific reasoning—understanding their capacity forself-correctionhas
become critically important.
Recent work has investigated whether LLMs can improve their outputs through self-refinement [Madaan
et al., 2023, Shinn et al., 2023]. While some studies report improvements through iterative prompting,
a growing body of evidence suggests thatintrinsicself-correction—where models correct errors without
external validation signals—is fundamentally limited [Huang et al., 2024].
In this paper, we argue that the concept of “self-correction” conflates several distinct capabilities that
deserve independent study. We propose adecomposition frameworkthat separates self-correction into
three measurable sub-capabilities (Figure 1):
1.Error Detection: Can the model identify that its output contains an error?
2.Error Localization: Can it pinpointwherethe error occurs?
3.Error Correction: Can it produce a corrected solution?
1
arXiv:2601.00828v1  [cs.AI]  24 Dec 2025

<!-- page 2 -->

Error
Detection
Error
Localization
Error
Correction
“Is it wrong?” “Where is the error?” “Can it be fixed?”
10–82% range Step identification 17–29% intrinsic
Figure 1: The Self-Correction Decomposition Framework. We separate self-correction into three distinct,
independently measurable capabilities.
By decomposing self-correction into these sub-capabilities, we reveal important andcounterintuitive
insights about the limitations of current LLMs. Our key contributions are:
• We propose a noveldecomposition frameworkfor analyzing LLM self-correction, separating detection,
localization, and correction into independently measurable capabilities.
• We discover theAccuracy-Correction Paradox: weaker models (GPT-3.5, Claude) achieve 1.6–1.7×
higher intrinsic correction rates than stronger models (DeepSeek).
• We propose theError Depth Hypothesis: stronger models make “deeper” errors that resist self-correction,
while weaker models make “shallower” errors that are easily fixable.
• We demonstrate thaterror detection is architecture-dependent, with Claude achieving only 10% detec-
tion vs 82% for GPT-3.5—an 8×difference.
2 Related Work
Self-Refinement in LLMs.Self-Refine [Madaan et al., 2023] proposed iterative refinement where models
generate, critique, and revise their outputs. Reflexion [Shinn et al., 2023] incorporated verbal reinforcement
learning for multi-step tasks. However, Huang et al. [2024] demonstrated that without external oracles,
LLMs cannot reliably self-correct reasoning errors—a finding our work extends with cross-model analysis.
Error Detection and Verification.Lightman et al. [2023] explored process reward models for verifying
intermediate reasoning steps. Cobbe et al. [2021] introduced verifiers for mathematical problem-solving.
Our work complements this by examining whether models can verifytheir ownoutputs and how this varies
across architectures.
Mathematical Reasoning.Chain-of-thought prompting [Wei et al., 2022] and related techniques have
substantially improved LLM performance on mathematical reasoning. GSM8K [Cobbe et al., 2021] has
become a standard benchmark for evaluating these capabilities.
3 Methodology
3.1 Problem Formulation
Given a mathematical reasoning problemP, an LLM generates a solutionS={s 1, s2, . . . , sn}consisting
ofnreasoning steps, with a final answerA. IfA̸=A ∗ (the gold answer), we sayScontains an error. We
decompose self-correction into three tasks:
2

<!-- page 3 -->

1.Error Detection: Given(P, S), predict whetherSis correct or incorrect.
2.Error Localization: Given(P, S)whereSis incorrect, identify the stepkwhere the first error occurs.
3.Error Correction: Given(P, S)and optionally the error locationk, produce a corrected solutionS ′ with
A′ =A ∗.
3.2 Experimental Setup
Dataset.We use GSM8K [Cobbe et al., 2021], a dataset of grade-school math word problems requiring
multi-step arithmetic reasoning. We sample problems using a fixed random seed (42) for reproducibility.
GSM8K-Complex (Ours).To rigorously test self-correction on capable models, we introduceGSM8K-
Complex, a subset of 500 problems filtered for higher complexity. We select problems meetingat least 2
of 3criteria: (1) question length>100 characters; (2) solution contains≥4 computation steps (counted by
“«” markers); (3) solution contains≥3 distinct operations. Problem indices will be released with code.
Models.We evaluate three models representing different capability levels and provider architectures:
•DeepSeek-Chat: A capable instruction-tuned model (94% baseline accuracy on GSM8K-Complex)
•GPT-3.5-Turbo: OpenAI’s efficient model (68% baseline accuracy)
•Claude-3-Haiku: Anthropic’s fast model (73.3% baseline accuracy)
Evaluation Protocol.For each model, we:
1. Generate solutions for all problems
2. Collect incorrect solutions (where model answer̸=gold answer)
3. TestError Detection: Does the model correctly classify its solution as incorrect?
4. TestCorrection with Hint: Given the error step location, can the model correct the solution?
5. TestIntrinsic Correction: Without hints, can the model correct the solution?
6. TestIterative Reflection: Over up to 3 rounds, can verification and re-solving succeed?
3.3 Metrics
•Detection Accuracy: Fraction of incorrect solutions correctly identified as incorrect
•Correction Success Rate: Fraction of incorrect solutions successfully corrected to the gold answer
•Iterative Success Rate: Maximum correction rate achieved across multiple reflection rounds
4 Results
4.1 Cross-Model Comparison
Table 1 presents the main findings across all three models.
3

<!-- page 4 -->

Table 1: Cross-Model Self-Correction Performance on GSM8K-Complex (n=500 per model)
Model Acc. Errors Detection Intrinsic With Hint Iterative
DeepSeek 94.0% 30 17/30 (56.7%) 5/30 (16.7%) 8/30 (26.7%) 6/30 (20.0%)
[37, 75] [6, 35] [12, 46] [8, 39]
GPT-3.5 66.4% 168 137/168 (81.5%) 45/168 (26.8%) 26/168 (15.5%) 114/168 (67.9%)
[75, 87] [20, 34] [10, 22] [60, 75]
Claude 70.4% 148 15/148 (10.1%) 43/148 (29.1%) 19/148 (12.8%) 90/148 (60.8%)
[6, 16] [22, 37] [8, 19] [53, 69]
Note: 95% Clopper-Pearson confidence intervals shown in brackets.Detection: correctly identifying error exists.
Intrinsic: correction without hints.Iterative: up to 3 reflection rounds.
4.2 The Accuracy-Correction Paradox
Figure 2 visualizes our key finding: the inverse relationship between model accuracy and intrinsic correction
capability.
DeepSeek GPT-3.5 Claude
0
20
40
60
80
100 94
66.4 70.4
16.7
26.8 29.1
Percentage (%)
Model Accuracy Intrinsic Correction
Figure 2: The Accuracy-Correction Paradox. Thestrongestmodel (DeepSeek, 94% accuracy) achieves the
lowestintrinsic correction rate (16.7%), while weaker models correct 1.6–1.7×more errors.
Finding 1: The Accuracy-Correction Paradox.The strongest model (DeepSeek, 94% accuracy) achieves
thelowestintrinsic correction rate (16.7%), while weaker models (GPT-3.5: 26.8%, Claude: 29.1%) cor-
rect 1.6–1.7×more errors. This suggests that model capability does not linearly translate to self-correction
ability.
Finding 2: Detection Does Not Predict Correction.GPT-3.5 detects 81.5% of errors but corrects only
26.8%. Claude detects only 10.1% but corrects 29.1%—achievinghighercorrection with 8×worsedetec-
tion. Detection and correction are largely independent capabilities.
Finding 3: Model-Generated Hints Hurt.Surprisingly, providing error location hints from the model’s
own localizationdecreasescorrection rates for all models (GPT-3.5: 26.8%→15.5%, Claude: 29.1%→12.8%).
4

<!-- page 5 -->

Since hints are model-generated (not ground-truth), this may reflect poor localization quality or anchoring
to incorrect reasoning paths.
Finding 4: Iterative Reflection Compensates for Weak Detection.Despite Claude’s 10% detection
rate, iterative reflection achieves 60.8% correction—a 6×improvement. Multi-round prompting bypasses
detection limitations through repeated re-solving.
4.3 Error Type Analysis
Table 2 reveals the distribution of error types, supporting our Error Depth Hypothesis.
Table 2: Error Type Distribution Across Models
Error Type DeepSeek GPT-3.5 Claude
Setup/Interpretation 44% 25% 38%
Logic Error 33% 13% 25%
Calculation Error 22%62%37%
Labeling Methodology.Error types are assigned automatically using the model’s own error localization
capability (see Appendix A.2 for prompt). Each model classifies its errors into: CALCULATION (arith-
metic mistakes), LOGIC (incorrect reasoning), or SETUP (problem misinterpretation). This approach pro-
vides scalable labeling but may underestimate certain error types if the model lacks self-awareness. We
acknowledge this as a limitation; future work should validate against human annotations.
A crucial pattern emerges: GPT-3.5’s errors are predominantlycalculation errors(62%), which are
typically “shallower” and easier to self-correct. DeepSeek’s errors are mostlysetup and logic errors(77%),
representing deeper reasoning failures that resist intrinsic correction.
4.4 Iterative Reflection Dynamics
Figure 3 shows how correction rates evolve across reflection rounds.
1 2 30
20
40
60
80
Reflection Round
Cumulative Correction Rate (%)
DeepSeek
GPT-3.5
Claude
Figure 3: Iterative Reflection Dynamics. DeepSeek saturates after round 1 (20%), while GPT-3.5 and Claude
continue improving through round 3 (68%, 61%).
5

<!-- page 6 -->

DeepSeek saturates after round 1 with minimal gains (20%), while GPT-3.5 and Claude continue im-
proving through round 3 (68%, 61%). This confirms that models with “shallower” errors benefit more from
iterative approaches.
5 Analysis and Discussion
5.1 The Error Depth Hypothesis
Why does a weaker model correct better? We propose theError Depth Hypothesis:
Stronger models make fewer but “deeper” errors (setup, logic) that are fundamentally harder
to self-correct. Weaker models make more but “shallower” errors (calculation) that are easily
fixable upon re-examination.
This hypothesis is supported by our error type analysis (Table 2): 62% of GPT-3.5 errors are calculation
mistakes, vs only 22% for DeepSeek. Calculation errors often involve simple arithmetic that the model can
correct when prompted to “check carefully.” Setup and logic errors require fundamental re-thinking of the
problem approach—something models struggle to do without external guidance.
5.2 Architecture-Dependent Detection
The dramatic variation in detection rates (10% to 82%) across architectures suggests that self-verification is
not a universal capability. Claude’s low detection rate despite reasonable correction via iteration indicates
that some models may “accidentally” correct errors through re-solving without explicitly recognizing them.
5.3 Implications for Self-Refinement Systems
Our findings have important practical implications:
1.Self-correction efficacy depends on error types.Systems should characterize the error distribution
before deploying self-correction strategies.
2.Stronger models may need different interventions.Model-generated hints hurt all models in our study,
suggesting localization-based feedback requires higher quality hints (e.g., human-annotated).
3.Iterative reflection is more valuable for weaker models.Multi-round prompting yields 3×improve-
ment for GPT-3.5/Claude but minimal gains for DeepSeek.
4.Detection̸=Correction across all models.Claude detects 10% but corrects 29%; GPT-3.5 detects 82%
but corrects 27%.
5.4 Limitations
Sample Size.While GSM8K-Complex yields 346 total errors (DeepSeek: 30, GPT-3.5: 168, Claude:
148), DeepSeek’s high accuracy still limits statistical power for that model. Future work should use even
harder benchmarks or synthetic error injection.
Three Models.Extending to additional models (GPT-4, Claude-Sonnet, Llama-3) would further validate
the accuracy-correction paradox.
6

<!-- page 7 -->

Hint Oracle.Our “with hint” condition uses model-generated step localization, not ground-truth annota-
tions. The surprising negative effect of hints warrants investigation with human-annotated error locations.
Domain Specificity.Mathematical reasoning has ground truth; findings may differ for open-ended gener-
ation tasks.
6 Conclusion
We have presented a decomposition framework for analyzing LLM self-correction, separating error detec-
tion, localization, and correction as distinct capabilities. Our experiments on GSM8K-Complex (n=500 per
model, 346 total errors) reveal the strikingAccuracy-Correction Paradox: the strongest model (DeepSeek)
achieves the lowest intrinsic correction rate (17%), while weaker models correct 1.6–1.7×more errors. We
propose theError Depth Hypothesis: stronger models make “deeper” errors that resist self-correction.
Surprisingly, providing error location hintshurtsall models, while iterative reflection compensates dra-
matically for weak detection (Claude: 10% detect→61% iterative). Our findings challenge linear as-
sumptions about model capability and self-improvement, with important implications for the design of self-
refinement pipelines.
7 Future Work
Future directions include: (1) scaling experiments to larger sample sizes across additional models including
GPT-4, Claude-Sonnet, and Llama-3; (2) investigating whether fine-tuning on self-correction data improves
intrinsic correction rates; (3) developing hybrid systems that combine model-based detection with tool-
augmented correction (e.g., calculators); and (4) extending the analysis to other reasoning domains including
code generation and commonsense reasoning.
Reproducibility Statement
All experiments use publicly available models accessed via API. We provide complete prompt templates
and evaluation scripts. Random seed is fixed at 42 for dataset sampling. Full experiment code and results
are available at:https://github.com/Kevin0304-li/llm-self-correction.
References
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word
problems.arXiv preprint arXiv:2110.14168, 2021.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word
problems.arXiv preprint arXiv:2110.14168, 2021.
Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and
Denny Zhou. Large language models cannot self-correct reasoning yet. InInternational Conference on
Learning Representations (ICLR), 2024.
7

<!-- page 8 -->

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John
Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step.arXiv preprint arXiv:2305.20050,
2023.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-
feedback. InAdvances in Neural Information Processing Systems (NeurIPS), 2023.
Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion:
Language agents with verbal reinforcement learning. InAdvances in Neural Information Processing
Systems (NeurIPS), 2023.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. Chain-of-thought prompting elicits reasoning in large language models. InAdvances in Neural
Information Processing Systems (NeurIPS), 2022.
A Prompt Templates
A.1 Error Detection Prompt
Look at this solution to a math problem. Is the solution correct?
Question: {question}
Solution: {solution}
Final Answer Given: {predicted_answer}
Analyze the solution carefully. Is there any error?
Respond with:
VERDICT: CORRECT or INCORRECT
CONFIDENCE: HIGH, MEDIUM, or LOW
EXPLANATION: (brief explanation)
A.2 Intrinsic Correction Prompt
Please verify your previous solution and correct any errors.
Question: {question}
Your previous solution: {solution}
Your previous answer: {predicted_answer}
Please carefully check each step. If you find any errors,
provide the corrected solution and final answer.
8

<!-- page 9 -->

Table 3: Detailed Experimental Configuration
Parameter Value
Dataset GSM8K (test split)
Random Seed 42
Temperature 0.0
Max Tokens 2048
Iterative Rounds 3
B Detailed Results
C GSM8K-Complex Problem IDs
The GSM8K-Complex (Ours) subset consists of 500 problems filtered from the GSM8K test set. We select
problems meetingat least 2 of 3criteria: (1) question length>100 characters; (2) solution contains≥4
computation steps (counted by “«” markers); (3) solution contains≥3 distinct mathematical operations.
The complete list of problem indices (0-indexed from the GSM8K test split) will be released with our code
repository.
9
