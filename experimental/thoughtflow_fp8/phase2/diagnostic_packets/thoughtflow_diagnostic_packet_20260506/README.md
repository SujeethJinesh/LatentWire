# ThoughtFlow Diagnostic Packet

Status: tracked falsification provenance packet, not positive-method evidence.

Run:

```bash
./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/build_diagnostic_packet.py \
  --output .debug/thoughtflow_diagnostic_packet_check
```

Expected pass condition: the builder exits successfully only from a clean
`experimental/thoughtflow_fp8/` path and rewrites the same manifest/table shape
with explicit hashes for the saved decision artifacts.

Then inspect:

- `manifest.json` for git state, source metadata, saved-artifact hashes, and
  tracked input-hash provenance for the saved diagnostic packet.
- `falsification_table.md` for the consumed signal ladder and stop decisions.
- `../../current_decision_manifest_20260506.md` for the current branch-level
  claim boundary.

This tracked packet lives outside ignored `results/` directories so
clean checkouts can run the saved-artifact tests. The manifest's git head is
the historical packet-generation commit; its script SHA is the current builder
file hash used by the verifier to detect drift.
This packet does not reopen the stopped RDU/PSI/VWAC branches.

Local correctness command:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 \
TRITON_HOME="$PWD/.debug/triton_home" \
./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -q -rs
```
