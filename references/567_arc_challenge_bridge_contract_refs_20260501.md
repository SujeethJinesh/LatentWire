# ARC-Challenge Bridge Contract References, 2026-05-01

## Role

This memo supports the first public benchmark bridge for the source-private
fixed-12B train-donor anti-shuffle packet. ARC-Challenge is a bridge and
leakage-control benchmark, not a final ICLR benchmark by itself.

## Primary Sources

- ARC / AI2 Reasoning Challenge:
  https://arxiv.org/abs/1803.05457
- TensorFlow Datasets `ai2_arc` card and official split counts:
  https://www.tensorflow.org/datasets/catalog/ai2_arc
- Hugging Face `allenai/ai2_arc` dataset:
  https://huggingface.co/datasets/allenai/ai2_arc
- ARC-DA direct-answer variant, useful for MCQ-artifact caveats:
  https://arxiv.org/abs/2102.03315
- EleutherAI Language Model Evaluation Harness ARC task, useful for standard
  evaluation framing:
  https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/arc
- Data contamination / evaluation malpractice caveat:
  https://arxiv.org/abs/2402.03927

## Boundary

Do not claim ARC SOTA or broad reasoning improvement from a small ARC bridge.
The defensible claim is narrower:

- official public multiple-choice benchmark bridge;
- fixed packet budget chosen before the public eval;
- label-blind source-packet construction;
- paired uncertainty against target-only and destructive controls;
- label-position and candidate-derangement audits to rule out answer-key
  shortcuts.

ARC alone is not enough for ICLR. It should be paired with at least one second
public benchmark or native systems evidence before a full-paper claim.
