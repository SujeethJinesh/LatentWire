# Locked Rate-Frontier Validation References, 2026-05-01

## Role

This memo supports the train-donor anti-shuffle locked budget-selection
artifact. The specific paper risk is reviewer suspicion that `12-14B` was
chosen after inspecting final eval rows.

## Primary Sources

- Elements of Statistical Learning:
  https://hastie.su.domains/ElemStatLearn/
- Deep Learning textbook:
  https://www.deeplearningbook.org/
- Deep Learning, Chapter 5 validation/test guidance:
  https://www.deeplearningbook.org/contents/ml.html
- On Over-fitting in Model Selection and Subsequent Selection Bias in
  Performance Evaluation:
  https://jmlr.org/beta/papers/v11/cawley10a.html
- Bias in error estimation when using cross-validation for model selection:
  https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-91
- Random Search for Hyper-Parameter Optimization:
  https://www.jmlr.org/papers/v13/bergstra12a.html
- A Survey of Cross-Validation Procedures for Model Selection:
  https://arxiv.org/abs/0907.4728
- A bias correction for the minimum error rate in cross-validation:
  https://arxiv.org/abs/0908.2904
- DomainBed:
  https://openreview.net/forum?id=lQdXeXDoWtI
- WILDS:
  https://arxiv.org/abs/2012.07421
- A principled approach to model validation in domain generalization:
  https://arxiv.org/abs/2304.00629
- Selective Classification for Deep Neural Networks:
  https://papers.nips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf
- Conformal Risk Control:
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
- End-to-end Optimized Image Compression:
  https://arxiv.org/abs/1611.01704
- Neural Rate Control for Learned Video Compression:
  https://openreview.net/forum?id=42lcaojZug
- KIVI:
  https://arxiv.org/abs/2402.02750
- KVQuant:
  https://arxiv.org/abs/2401.18079

## Boundary For This Paper

Safe framing: byte budget is a hyperparameter/rate point. It can be selected on
a validation frontier, then frozen before final eval. The paper should report
the validation curve, the frozen selected row, and adjacent sensitivity rows.

Unsafe framing: choose the best byte budget after reading final n512 eval, then
present only that row as the method.

The current artifact is useful but incomplete for ICLR: it validates a
per-seed frontier (`47 -> 14B`, `53/59 -> 12B`) and all selected n512 rows pass.
The train-family disjoint follow-up makes the remaining selector risk more
specific:

- all-controls validation fails because matched-byte structured text is too
  strong on the holdout-family validation side;
- source-private-controls-only validation selects `10B`, but that selected
  budget fails the n512 holdout-to-core CI margin against the learned-synonym
  base.
- a stable-interior validation selector is a more conservative alternative:
  select the smallest validation-clean budget that has clean neighboring rates
  on both sides, then report adjacent rates as sensitivity. This follows the
  model-selection lesson that minimum validation estimates can be optimistic
  under finite samples and multiple candidate rates.

The cleaner final submission needs one of:

- a global fixed budget that clears all seeds;
- a predeclared rate band with all rates reported;
- or an example-level diagnostic budget selector that is trained/validated
  without final eval labels.
