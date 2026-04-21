| Method | Candidate Acc | Baseline Acc | Delta | Cand Only | Base Only | 95% Bootstrap Delta | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_prior | 0.0857 | 0.0429 | +0.0429 | 3 | 0 | [+0.0000, +0.1000] | 0.2482 |
| grouped_subspace_resid4 | 0.0571 | 0.0857 | -0.0286 | 0 | 2 | [-0.0714, +0.0000] | 0.4795 |
| bridge_ridge | 0.0429 | 0.0571 | -0.0143 | 3 | 4 | [-0.0857, +0.0571] | 1.0000 |
| bridge_ridge | 0.0429 | 0.0857 | -0.0429 | 3 | 6 | [-0.1286, +0.0429] | 0.5050 |
| grouped_subspace_resid4 vs c2c | 0.0571 | 0.1286 | -0.0714 | 4 | 9 | [-0.1714, +0.0286] | 0.2673 |
| fixed_prior vs c2c | 0.0857 | 0.1286 | -0.0429 | 5 | 8 | [-0.1429, +0.0571] | 0.5791 |
| bridge_ridge_control vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| grouped_rotational_transport vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| grouped_fitted_rotation_transport vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| grouped_shared_basis_transport vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| shared_plus_private_asym_adapter vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| shared_plus_private_dynmap_adapter vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| xattn_adapter vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| xattn_dynmap_adapter vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| module_adapter vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| spanalign_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| bytespan_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_ctxonly_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_dwakd_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_likelihood_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_spanalm_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_dwainteract_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_prefdist_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_prefdist_attention_stratified vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| dynalign_prefdist_query_pool_transport vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| readout_adapter vs target_alone_control | 0.0000 | 0.1000 | -0.1000 | 0 | 1 | [-0.3000, +0.0000] | 1.0000 |
| dynalign_interact_module_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
| tokenbasis_replace vs target_alone_control | 0.1000 | 0.1000 | +0.0000 | 0 | 0 | [+0.0000, +0.0000] | 1.0000 |
