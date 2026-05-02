# Systems Boundary Figure/Table V3

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM workshop is stronger after this update; ICLR
  full paper is still blocked by the missing cross-family positive connector
  and native NVIDIA serving rows.
- Current story: LatentWire has a clean source-private systems boundary:
  fixed-byte task packets on ARC/OpenBookQA/HellaSwag sit far left of
  KV/cache-state transfer while keeping source text, source KV, and source
  hidden vectors out of the communicated object.
- Exact remaining blocker: the artifact supports byte/exposure accounting, not
  a native C2C/KVComm/TurboQuant/QJL/vLLM/SGLang speed or quality win.

## Lay Explanation

This table asks how much information crosses the boundary. LatentWire sends a
tiny task hint. KV/cache systems send or store internal model memory. Even very
optimistic one-token KV sketches are much larger than the packet, but that is a
byte-accounting result, not proof that LatentWire is faster on a GPU.

## Artifact

`results/source_private_systems_boundary_figure_table_20260502/`

Files:

- `systems_boundary_figure_data.json`
- `systems_boundary_figure_data.csv`
- `systems_boundary_table.md`
- `systems_boundary_table.tex`
- `systems_boundary_waterfall.svg`
- `manifest.json`

## Result

- pass gate: `True`
- packet rows: `4`
- packet framed-byte range: `4-15B`
- minimum source-state floor: `768B`
- minimum source-state floor versus largest packet: `51.2x`
- native NVIDIA systems complete: `False`

Key rows:

| Row | Object | Framed bytes | Source text | Source KV | Source hidden | Status |
|---|---|---:|---:|---:|---:|---|
| LatentWire ARC-Challenge | task-level candidate evidence packet | `15B` | no | no | no | Mac-local artifact |
| LatentWire OpenBookQA | task-level candidate evidence packet | `6B` | no | no | no | Mac-local artifact |
| LatentWire HellaSwag compact | task-level candidate evidence packet | `4B` | no | no | no | Mac-local artifact |
| QJL sign-bit KV floor | one-token K+V state | `768B` | no | yes | no | byte floor, not native |
| TurboQuant 3.5-bit KV floor | one-token K+V state | `2688B` | no | yes | no | byte floor, not native |
| KVComm 30% fp16 floor | selected source KV layers | `3686.4B` | no | yes | no | byte floor, not native |
| C2C fp16 floor | projected/fused source KV cache | `12288B` | no | yes | no | byte floor, not native |

## Interpretation

This is now the safest systems-side win for the current paper. It does not
claim faster serving. It claims that the LatentWire communication object is a
fixed-byte source-private packet, while direct cache communication and KV
quantization baselines operate on source-state objects.

The strongest reviewer-safe sentence is:

> Across the current Mac-local packet rows, LatentWire communicates `4-15B`
> framed task packets with no source text, source KV, or source hidden-vector
> exposure; conservative one-token KV/source-state byte floors start at `768B`,
> and native serving baselines remain required for latency and goodput claims.

## Decision

Promote this as the paper-ready systems boundary figure/table. Use the TeX
table and SVG in COLM/ICLR drafts as an accounting figure, while keeping native
NVIDIA systems rows as an explicit blocker.

## Next Exact Gate

Run a stronger true cross-family source or trainable query/cache connector on
NVIDIA. For systems, fill the native schema with matched vLLM/SGLang/C2C/KVComm
rows reporting TTFT, TPOT, goodput, GPU memory, HBM/PCIe/NVLink bytes, accuracy,
payload/framed bytes, and source-exposure flags.
