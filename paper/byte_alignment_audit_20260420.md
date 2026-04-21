# Byte Alignment Audit

- prompts: `8`
- changed prompts: `1`
- mean span pairs: `21.62`
- mean byte pairs: `21.62`

| Changed | UTF-8 bytes | Src toks | Tgt toks | Span pairs | Byte pairs | Prompt |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 51 | 64 | 36 | 24 | 24 | Compute 7½% of $1,234.56, then add 3 km in meters. |
| 0 | 68 | 63 | 35 | 23 | 23 | If a café sells 12 croissants at €2.50 each, what is the revenue? |
| 0 | 44 | 61 | 33 | 21 | 21 | Solve: α + β = 17, and β = 5. What is α? |
| 1 | 73 | 61 | 33 | 21 | 21 | Emoji check: 🧪 + 🧠 = one combined clue. What two objects are shown? |
| 0 | 62 | 64 | 36 | 24 | 24 | Chemistry tokenization: NaCl, H₂O, CO₂, and 10⁻³ mol/L. |
| 0 | 69 | 60 | 32 | 20 | 20 | Code stress: for i in range(3): total += nums[i]. What is loop count? |
| 0 | 77 | 58 | 30 | 18 | 18 | Multilingual stress: 東京 to Paris is written as Tokyo to Paris in English. |
| 0 | 58 | 62 | 34 | 22 | 22 | Units stress: 5 µs + 20 ms + 3 ns; which unit is largest? |
