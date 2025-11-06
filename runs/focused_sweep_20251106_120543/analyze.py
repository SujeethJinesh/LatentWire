import json
import glob
import os

sweep_dir = os.path.dirname(os.path.abspath(__file__))
results = []

for exp_dir in glob.glob(os.path.join(sweep_dir, "*/")):
    if os.path.isdir(exp_dir):
        name = os.path.basename(exp_dir.rstrip("/"))
        log_file = os.path.join(exp_dir, "train.log")

        if os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()

            # Find all eval lines
            evals = []
            for line in lines:
                if "Eval" in line and "Target-alone" in line:
                    parts = line.split("|")
                    step = int(parts[0].split()[-1])
                    target = float(parts[1].split(":")[-1])
                    bridged = float(parts[2].split(":")[-1])
                    evals.append({"step": step, "target": target, "bridged": bridged})

            if evals:
                results.append({
                    "name": name,
                    "evals": evals,
                    "final_target": evals[-1]["target"],
                    "final_bridged": evals[-1]["bridged"],
                    "max_bridged": max(e["bridged"] for e in evals),
                    "max_bridged_step": max(evals, key=lambda e: e["bridged"])["step"]
                })

# Sort by final bridged accuracy
results.sort(key=lambda x: x["final_bridged"], reverse=True)

print("\n=== Best Final Bridged Accuracy ===")
for r in results[:3]:
    print(f"{r['name']}: {r['final_bridged']:.1%} (target: {r['final_target']:.1%})")

print("\n=== Best Peak Bridged Accuracy ===")
results.sort(key=lambda x: x["max_bridged"], reverse=True)
for r in results[:3]:
    print(f"{r['name']}: {r['max_bridged']:.1%} at step {r['max_bridged_step']}")

# Save as JSON
with open(os.path.join(sweep_dir, "analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to: {os.path.join(sweep_dir, 'analysis.json')}")
