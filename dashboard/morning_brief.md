# Morning Brief

Stage 0 cache freeze ran locally on the Mac in `metadata_file_level` mode. It found `1087` cache units and `8725` row/file entities. `769` units lack a clean named dev/gate/confirm partition or look like prior eval/full-test caches, so any screen using them is provisional.

Next: build/run Stage-1 screens only against dev/gate split manifests, then park all confirmation/GPU work in `queues/parked.yaml`.
