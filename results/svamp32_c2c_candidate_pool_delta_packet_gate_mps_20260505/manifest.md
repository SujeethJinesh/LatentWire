# SVAMP32 C2C Candidate-Pool Delta Packet Gate Manifest

- date: `2026-05-05`
- status: `c2c_candidate_pool_delta_packet_capacity_fails_controls`
- output json: `results/svamp32_c2c_candidate_pool_delta_packet_gate_mps_20260505/candidate_pool_delta_packet_gate.json`
- output md: `results/svamp32_c2c_candidate_pool_delta_packet_gate_mps_20260505/candidate_pool_delta_packet_gate.md`
- manifest json: `results/svamp32_c2c_candidate_pool_delta_packet_gate_mps_20260505/manifest.json`

## Interpretation

- This artifact removes teacher-generated-prefix conditioning and scores numeric candidates directly.
- It is still a dense-teacher capacity gate because packet values are computed from C2C candidate scores.
- If matched fails against candidate-roll or wrong-row controls, the packet is not a source-causal method target.
