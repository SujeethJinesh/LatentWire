# Source-Private Test-Log Packet Strict-Small Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_strict_small.py --examples 160 --candidates 4 --seed 28 --budgets 2,4,8,16,32 --output-dir results/source_private_testlog_packet_strict_small_20260428
```

## Outcome

- strict-small pass: `True`
- passing budgets: `[2, 4, 8, 16, 32]`
- best budget bytes: `2`

## Artifacts

- `benchmark.jsonl`
- `sweep_summary.json`
- `sweep_summary.md`
- `leakage_audit.json`
- `leakage_audit.md`
- `manifest.json`
- `manifest.md`
- `predictions_budget2.jsonl`
- `summary_budget2.json`
- `predictions_budget4.jsonl`
- `summary_budget4.json`
- `predictions_budget8.jsonl`
- `summary_budget8.json`
- `predictions_budget16.jsonl`
- `summary_budget16.json`
- `predictions_budget32.jsonl`
- `summary_budget32.json`

## Artifact Hashes

- `benchmark.jsonl`: `7c9e4bffe9686e0c5eafdc4e0d5795173aaf771201a217853254785d9a7696c5`
- `leakage_audit.json`: `49a8bb4dc42be0ee9e9cb15ed45546bf7bbd6b19c1dc4883e00782c09ca5324d`
- `leakage_audit.md`: `c580bd7c20be9e5aedc5c5c3369a25846bc9afebb054fe3d197e13e0a969e2e5`
- `predictions_budget16.jsonl`: `b6d9183bf094d85fd2b46ce7aded1f77f0e3437043da4727c7738da9ff016778`
- `predictions_budget2.jsonl`: `668191464577c934f9d8a0252a7bcc8a8fd54874ee073df4a3ccd1b3d052a0ea`
- `predictions_budget32.jsonl`: `d358d0b05f60a053214939aae8e42fb4e4d7e25ece4dbd02c572487037b7622d`
- `predictions_budget4.jsonl`: `c3c41cc1e70b5788cf9b86fcccb1b9502ebf4a2dedc97e9132babb9e7a553eb2`
- `predictions_budget8.jsonl`: `2f633a2c91f200daeabacb9ffa3b2b42b02b63afd4448482f5f2b46694d16ec7`
- `summary_budget16.json`: `931b74515633576c8f852ac3c6ce2fd181731ded3ef74dd1de5d19c77084d445`
- `summary_budget2.json`: `5d0038601ee040b3c07e7ede2bf841d11eac83a867d53f898d9d20adeaf715c7`
- `summary_budget32.json`: `602905500c15d0a0a9f2454caa0f69db51e59b1bd051ad0d5309889ca564af51`
- `summary_budget4.json`: `c05d946b6d52f3e445322a3aaca3c2bb1a0473100b2631c8b4fb3d5c0298b1b2`
- `summary_budget8.json`: `67b8d271d1024ffb10372128623026eb738f912192a777daa3269d565d4faf49`
- `sweep_summary.json`: `7d6f7ad51e17d7393641adb73243264791d84309fbf94853f640f70f99bb8c60`
- `sweep_summary.md`: `652e5c07cb22e9d9d253d6b1aaeb347fcdb420af3a47bbc20c97472a018d4a72`