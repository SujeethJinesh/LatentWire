.PHONY: setup-mac setup-linux smoke train eval

setup-mac:
	bash scripts/setup_mac.sh

setup-linux:
	bash scripts/setup_linux.sh

smoke:
	bash scripts/run_smoke_cpu.sh

train:
	bash scripts/run_train_small.sh

eval:
	bash scripts/run_eval_small.sh

.PHONY: prefetch
prefetch:
	bash scripts/prefetch_assets.sh "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "Qwen/Qwen2-0.5B-Instruct"
