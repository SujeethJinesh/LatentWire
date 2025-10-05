#!/usr/bin/env python3
"""
Analyze diagnostics to find first F1 > 0 breakthrough moment.

Usage:
    python scripts/analyze_breakthrough.py runs/smoke_stageb_ext/diagnostics.jsonl
"""

import json
import sys
from pathlib import Path

def analyze_diagnostics(diag_path: str):
    """Find breakthrough moments in training diagnostics."""

    if not Path(diag_path).exists():
        print(f"Error: {diag_path} not found")
        return

    print("=" * 80)
    print("BREAKTHROUGH ANALYSIS")
    print("=" * 80)

    breakthroughs = []
    all_steps = []
    peak_first_acc = 0
    peak_step = 0

    with open(diag_path) as f:
        for line in f:
            data = json.loads(line.strip())
            step = data['global_step']
            epoch = data['epoch']
            mode = data['models']['llama']['mode']

            first_acc = data['models']['llama']['first_acc']
            first_loss = data['models']['llama']['first']
            kce = data['models']['llama']['kce']
            kd = data['models']['llama']['kd']
            tf = data['models']['llama']['tf']
            grad_norm = data['grad_norm']

            all_steps.append({
                'step': step,
                'epoch': epoch,
                'mode': mode,
                'first_acc': first_acc,
                'first_loss': first_loss,
                'kce': kce,
                'kd': kd,
                'tf': tf,
                'grad_norm': grad_norm,
            })

            # Track peak first_acc
            if first_acc > peak_first_acc:
                peak_first_acc = first_acc
                peak_step = step

            # Track breakthroughs (first_acc > 0)
            if first_acc > 0:
                breakthroughs.append({
                    'step': step,
                    'epoch': epoch,
                    'mode': mode,
                    'first_acc': first_acc,
                    'first_loss': first_loss,
                    'kce': kce,
                })

    # Print summary
    print(f"\nTotal steps: {len(all_steps)}")
    print(f"Peak first_acc: {peak_first_acc:.3%} at step {peak_step}")
    print(f"Breakthrough moments (first_acc > 0): {len(breakthroughs)}")

    # Print all breakthroughs
    if breakthroughs:
        print("\n" + "=" * 80)
        print("BREAKTHROUGH TIMELINE")
        print("=" * 80)
        for bt in breakthroughs:
            print(f"Step {bt['step']:3d} (epoch {bt['epoch']}, mode={bt['mode']:6s}): "
                  f"first_acc={bt['first_acc']:6.3%} | first={bt['first_loss']:5.2f} | kCE={bt['kce']:5.2f}")

    # Analyze regression if peak dropped
    final_step = all_steps[-1]
    if final_step['first_acc'] < peak_first_acc * 0.8:  # >20% drop
        print("\n" + "=" * 80)
        print("⚠️  REGRESSION DETECTED")
        print("=" * 80)
        print(f"Peak: {peak_first_acc:.3%} at step {peak_step}")
        print(f"Final: {final_step['first_acc']:.3%} at step {final_step['step']}")
        print(f"Drop: {(peak_first_acc - final_step['first_acc']):.3%} "
              f"({(1 - final_step['first_acc']/peak_first_acc)*100:.1f}% regression)")

        # Find when regression started
        for i, s in enumerate(all_steps):
            if s['step'] == peak_step and i + 1 < len(all_steps):
                print(f"\nRegression timeline:")
                for j in range(i, min(i + 10, len(all_steps))):
                    step_data = all_steps[j]
                    print(f"  Step {step_data['step']:3d}: first_acc={step_data['first_acc']:.3%} "
                          f"grad_norm={step_data['grad_norm']:6.2f}")
                break

    # Loss trends
    print("\n" + "=" * 80)
    print("LOSS TRENDS (every 40 steps)")
    print("=" * 80)
    for i, s in enumerate(all_steps):
        if s['step'] % 40 == 0 or i == len(all_steps) - 1:
            print(f"Step {s['step']:3d}: first_acc={s['first_acc']:.3%} | "
                  f"first={s['first_loss']:5.2f} | kCE={s['kce']:5.2f} | "
                  f"KD={s['kd']:5.2f} | tf={s['tf']:5.2f} | grad={s['grad_norm']:6.2f}")

    # Acceptance criteria
    print("\n" + "=" * 80)
    print("ACCEPTANCE CRITERIA")
    print("=" * 80)

    criteria = {
        "First-token learning": (peak_first_acc > 0, f"peak={peak_first_acc:.3%} (target: >0%)"),
        "First-token stability": (final_step['first_acc'] > peak_first_acc * 0.8,
                                 f"final={final_step['first_acc']:.3%} vs peak={peak_first_acc:.3%} (target: <20% drop)"),
        "Gradient health": (final_step['grad_norm'] < 500,
                          f"final_grad={final_step['grad_norm']:.2f} (target: <500)"),
        "First-token convergence": (peak_first_acc >= 0.12,
                                    f"peak={peak_first_acc:.3%} (target: ≥12%)"),
    }

    for name, (passed, detail) in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} | {name:30s} | {detail}")

    # Next steps recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if peak_first_acc >= 0.12 and final_step['first_acc'] > peak_first_acc * 0.8:
        print("✅ HERO-READY: First-token learning stable and converged (≥12%)")
        print("   Next: Run hero configuration (bash scripts/run_llama_single.sh --hero)")
    elif peak_first_acc >= 0.08 and final_step['first_acc'] > peak_first_acc * 0.8:
        print("⚠️  MARGINAL: First-token learning started (8-12%) but below target")
        print("   Next: Extend to 12+ epochs or increase FIRST_TOKEN_CE_WEIGHT_STAGEB")
    elif peak_first_acc > 0 and final_step['first_acc'] < peak_first_acc * 0.8:
        print("⚠️  REGRESSION: Learning started but unstable")
        print("   Next: Add learning rate scheduling (cosine decay from 5e-5 to 1e-6)")
    elif peak_first_acc > 0:
        print("⚠️  WEAK SIGNAL: Learning detected but very low")
        print("   Next: Increase EPOCHS_STAGEB to 12-16 or FIRST_TOKEN_CE_WEIGHT to 12-15")
    else:
        print("❌ NO LEARNING: Architecture issue detected")
        print("   Next: Implement scheduled sampling or check LoRA engagement")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_breakthrough.py <diagnostics.jsonl>")
        sys.exit(1)

    analyze_diagnostics(sys.argv[1])
