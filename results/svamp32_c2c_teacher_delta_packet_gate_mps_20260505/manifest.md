# SVAMP32 C2C Teacher-Delta Packet Gate Manifest

- date: `2026-05-05`
- status: `teacher_delta_packet_capacity_fails_controls`
- output json: `results/svamp32_c2c_teacher_delta_packet_gate_mps_20260505/teacher_delta_packet_gate.json`
- output md: `results/svamp32_c2c_teacher_delta_packet_gate_mps_20260505/teacher_delta_packet_gate.md`
- manifest json: `results/svamp32_c2c_teacher_delta_packet_gate_mps_20260505/manifest.json`

## Interpretation

- This artifact tests whether sparse top-k logit deltas from the dense C2C teacher survive destructive controls.
- The gate is not source-private because the packet is computed from dense C2C teacher logits.
- If matched does not beat target-only under the teacher-generated prefix, the apparent signal is a teacher-prefix/target-cache effect.
