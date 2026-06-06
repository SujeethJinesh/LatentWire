# Channel-Set Regime Checklist

| Requirement for a positive Channel-Set method | Current status | Paper treatment |
| --- | --- | --- |
| Static channel set is stable enough to protect | Fails as a general assumption: top-1 strict set-leaving is `0.533713` to `0.673611` across Granite-Small, Nemotron-3, DeepSeek-R1-Distill, and Falcon-H1 | Main measurement claim |
| Paired ParoQuant/static/EMA baseline lock | Not complete for C-A1 native same-row matrix | Limitation and future requirement |
| Fresh split-clean Granite/DeepSeek/Falcon C-A1 matrix | Missing; Granite same-row count is `0` in the safe manifest | C-A1 parked |
| No-gap denominator and OSC/DecDEC defense | Defense scaffold exists but is underpowered or missing native pairing; C-U1/C-S1/C-Y5 blocked | Blocker map, not completed defense proof |
| Held-out positive confirmation | Not authorized and not run | Explicitly unsupported |
| Claimable systems card | Missing native W4A16/ParoQuant replay | Future work |
