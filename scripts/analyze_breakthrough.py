#!/usr/bin/env python3
"""
Comprehensive training analysis: diagnostics, logs, and F1 breakthrough detection.

Usage:
    python scripts/analyze_breakthrough.py runs/smoke_stageb_ext/diagnostics.jsonl [--log pipeline.log] [--watch]

Options:
    --log: Also analyze pipeline log for F1 scores and peak detections
    --watch: Monitor logs in real-time (Ctrl+C to stop)
"""

import json
import sys
import re
import time
from pathlib import Path
from typing import Optional, Dict, List

def analyze_pipeline_log(log_path: str) -> Dict:
    """Extract F1 scores and training peaks from pipeline log."""
    if not Path(log_path).exists():
        return {}

    results = {
        'f1_scores': [],
        'peaks': [],
        'final_metrics': {},
    }

    with open(log_path) as f:
        for line in f:
            # Extract F1 scores
            if 'F1:' in line and 'Llama' in line:
                match = re.search(r'F1[=:]\s*([0-9]+\.[0-9]+)', line)
                if match:
                    f1 = float(match.group(1))
                    results['f1_scores'].append(f1)

                    # Extract EM if available
                    em_match = re.search(r'EM[=:]\s*([0-9]+\.[0-9]+)', line)
                    if em_match and 'latent' in line.lower():
                        results['final_metrics']['latent_em'] = float(em_match.group(1))
                        results['final_metrics']['latent_f1'] = f1
                    elif em_match and 'text' in line.lower():
                        results['final_metrics']['text_em'] = float(em_match.group(1))
                        results['final_metrics']['text_f1'] = f1

            # Extract NEW PEAK messages (EMA tracking)
            if 'NEW PEAK' in line:
                match = re.search(r'first_acc_ema=([0-9]+\.[0-9]+)%.*raw_batch=([0-9]+\.[0-9]+)%.*step (\d+)', line)
                if match:
                    results['peaks'].append({
                        'ema': float(match.group(1)),  # Already in percentage (5.2 = 5.2%)
                        'raw': float(match.group(2)),  # Already in percentage (12.5 = 12.5%)
                        'step': int(match.group(3)),
                    })

    return results

def analyze_diagnostics(diag_path: str, log_path: Optional[str] = None, watch: bool = False):
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

    # Log analysis (if provided)
    if log_path:
        print("\n" + "=" * 80)
        print("PIPELINE LOG ANALYSIS")
        print("=" * 80)

        log_results = analyze_pipeline_log(log_path)

        # Peak detection (EMA tracking)
        if log_results.get('peaks'):
            print("\nPeak Detection (EMA Tracking):")
            for peak in log_results['peaks']:
                print(f"  Step {peak['step']:3d}: EMA={peak['ema']:.1f}% | Raw batch={peak['raw']:.1f}%")

            max_raw = max(p['raw'] for p in log_results['peaks'])
            max_ema = max(p['ema'] for p in log_results['peaks'])
            print(f"\n  Peak raw batch: {max_raw:.1f}% {'✅ ≥12%!' if max_raw >= 12 else '❌ <12%'}")
            print(f"  Peak EMA: {max_ema:.1f}% (smoothed average)")

        # F1 scores
        if log_results.get('f1_scores'):
            f1_scores = log_results['f1_scores']
            f1_nonzero = [f for f in f1_scores if f > 0]

            print(f"\nF1 Scores Found: {len(f1_scores)}")
            if f1_nonzero:
                print(f"  🎉 F1 > 0 BREAKTHROUGH: {len(f1_nonzero)} scores above 0!")
                print(f"  Max F1: {max(f1_nonzero):.4f}")
                print(f"  First F1 > 0 at index: {f1_scores.index(f1_nonzero[0])}")
            else:
                print(f"  ❌ No F1 > 0 detected (all scores: {set(f1_scores)})")

        # Final metrics
        if log_results.get('final_metrics'):
            metrics = log_results['final_metrics']
            print("\nFinal Eval Metrics:")
            if 'text_em' in metrics:
                print(f"  Text baseline: EM={metrics['text_em']:.3f} F1={metrics['text_f1']:.3f}")
            if 'latent_em' in metrics:
                print(f"  Latent: EM={metrics['latent_em']:.3f} F1={metrics['latent_f1']:.3f}")

    # Next steps recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Check if we have peak info from logs (raw batch can be higher than diagnostics)
    max_raw_from_log = 0
    if log_path and log_results.get('peaks'):
        max_raw_from_log = max(p['raw'] for p in log_results['peaks']) / 100  # Convert % to fraction

    # Use the higher of diagnostics peak or log raw peak
    effective_peak = max(peak_first_acc, max_raw_from_log)

    if effective_peak >= 0.12:
        if max_raw_from_log >= 0.12 and peak_first_acc < 0.12:
            print("🎉 BREAKTHROUGH: Raw batch accuracy ≥12% (hero threshold!)")
            print("   ⚠️  BUT: Unstable (diagnostics peak only {:.1%})".format(peak_first_acc))
            print("   Next: Add learning rate scheduling (cosine decay) to stabilize")
            print("   Then: Run hero with LR schedule (expected: stable 12-20% first_acc)")
        else:
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
        print("Usage: python scripts/analyze_breakthrough.py <diagnostics.jsonl> [--log pipeline.log] [--watch]")
        sys.exit(1)

    diag_path = sys.argv[1]
    log_path = None
    watch = False

    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--log' and i + 1 < len(sys.argv):
            log_path = sys.argv[i + 1]
        elif arg == '--watch':
            watch = True

    # Auto-detect log file if in same directory
    if not log_path:
        diag_dir = Path(diag_path).parent
        log_files = list(diag_dir.glob('pipeline_*.log'))
        if log_files:
            log_path = str(log_files[0])

    analyze_diagnostics(diag_path, log_path, watch)
