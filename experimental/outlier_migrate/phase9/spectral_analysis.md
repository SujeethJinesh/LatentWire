# Phase 9 Spectral Analysis of Decode-Position Channel Trajectories

Generated: 2026-05-18

## Identifiability

The requested four-model, all-20K FFT is not fully identifiable from current packets. Granite-Small has a dense M11 packet sampled every 100 decode positions from 100 to 10000. The four-model packets for Granite-Small, Nemotron-3, DeepSeek-R1-Distill, and Falcon-H1 are sparse grids with 6 positions, so FFT and autocorrelation estimates would be misleading.

This report therefore computes the spectral readout only where the packet is dense and uniformly spaced, and records skipped packets explicitly.

## Dense-Packet Results

| Packet | Positions | Layers | Top-1% channels | Low-frequency power (first 10% bins) | Spectral entropy | Autocorr length (tokens) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite-Small dense M11 packet | 100 | 40 | 1640 | 0.274561 | 0.853098 | 100.000000 |

## Skipped Packets

| Packet | Positions | Reason |
| --- | ---: | --- |
| Granite-Small Phase 1 packet | 6 | FFT/autocorrelation requires dense uniformly spaced trajectories; this packet is sparse. |
| Nemotron-3-Nano Phase 2 packet | 6 | FFT/autocorrelation requires dense uniformly spaced trajectories; this packet is sparse. |
| DeepSeek-R1-Distill-Qwen-1.5B Phase 5' packet | 6 | FFT/autocorrelation requires dense uniformly spaced trajectories; this packet is sparse. |
| Falcon-H1 Phase 7 packet | 6 | FFT/autocorrelation requires dense uniformly spaced trajectories; this packet is sparse. |

## Interpretation

The dense Granite readout provides an initial sanity check, not a four-model conclusion. If the low-frequency power fraction is high and spectral entropy is low, streaming subspace or predictor methods are more plausible on Granite. If the spectrum is broad, local trajectory prediction is unlikely to rescue the current W4A16 protection family.

Because the dense packet stops at decode position 10000 and only covers Granite-Small, this analysis should not be used to claim cross-model spectral structure. A future component/dense packet would need uniform trajectories to 20000 on Nemotron, DeepSeek, and Falcon before promoting Streaming-PCA beyond a conditional follow-up.
