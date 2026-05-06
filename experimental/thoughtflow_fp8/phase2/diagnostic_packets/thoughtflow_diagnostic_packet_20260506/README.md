# ThoughtFlow Diagnostic Packet

Run:

```bash
./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/build_diagnostic_packet.py
```

Then inspect `manifest.json` and `falsification_table.md`.
This tracked packet lives outside ignored `results/` directories so
clean checkouts can run the saved-artifact tests.
This packet does not reopen the stopped RDU/PSI/VWAC branches.
