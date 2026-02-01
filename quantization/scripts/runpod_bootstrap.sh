#!/usr/bin/env bash
set -euo pipefail

INSTALL=0
FORCE=0
GENERATE_SSH=0
WRITE_ENV=0
WRITE_TMUX=0
while [ $# -gt 0 ]; do
  case "$1" in
    --full)
      INSTALL=1
      GENERATE_SSH=1
      WRITE_ENV=1
      WRITE_TMUX=1
      FORCE=1
      ;;
    --install) INSTALL=1 ;;
    --force) FORCE=1 ;;
    --generate-ssh-key) GENERATE_SSH=1 ;;
    --write-env) WRITE_ENV=1 ;;
    --write-tmux-conf) WRITE_TMUX=1 ;;
    --help|-h)
      echo "Usage: runpod_bootstrap.sh [--full] [--install] [--force] [--generate-ssh-key] [--write-env] [--write-tmux-conf]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

ROOT="${PROJECT_ROOT:-$(pwd)}"
RUNPOD_HINT=0
if [ -d "/workspace" ]; then
  RUNPOD_HINT=1
fi
if [ "${FORCE}" = "1" ]; then
  RUNPOD_HINT=1
fi

if [ "${RUNPOD_HINT}" != "1" ]; then
  echo "RunPod bootstrap: /workspace not found; skipping RunPod-only checks."
  exit 0
fi

fail=0
err() {
  echo "ERROR: $*" >&2
  fail=1
}

warn() {
  echo "WARN: $*" >&2
}

APT_UPDATED=0
apt_update_once() {
  if [ "${APT_UPDATED}" -eq 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
    else
      apt-get update
    fi
    APT_UPDATED=1
  fi
}

apt_install() {
  local pkg="$1"
  if ! command -v apt-get >/dev/null 2>&1; then
    err "apt-get is not available; install ${pkg} manually."
    return
  fi
  apt_update_once
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get install -y "${pkg}"
  else
    apt-get install -y "${pkg}"
  fi
}

if [ ! -f "/workspace/env.sh" ]; then
  if [ "${WRITE_ENV}" = "1" ]; then
    cat > /workspace/env.sh <<'EOF'
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export C2C_CKPT_ROOT=/workspace/c2c_checkpoints
export CONDA_EXE=/workspace/conda/bin/conda
export HF_TOKEN=YOUR_TOKEN_HERE
export WANDB_DISABLED=true
EOF
    echo "Wrote /workspace/env.sh (update HF_TOKEN)."
  else
    err "/workspace/env.sh missing."
    cat >&2 <<'EOF'
Create it with:
  cat > /workspace/env.sh <<'EOT'
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export C2C_CKPT_ROOT=/workspace/c2c_checkpoints
export CONDA_EXE=/workspace/conda/bin/conda
export HF_TOKEN=YOUR_TOKEN_HERE
export WANDB_DISABLED=true
EOT
EOF
  fi
fi

if [ -f "/workspace/env.sh" ]; then
  # shellcheck disable=SC1091
  source /workspace/env.sh || true
  if ! grep -qxF 'source /workspace/env.sh' "${HOME}/.bashrc" 2>/dev/null; then
    echo 'source /workspace/env.sh' >> "${HOME}/.bashrc"
    echo "Added /workspace/env.sh to ~/.bashrc."
  fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
  err "HF_TOKEN is not set. Export HF_TOKEN and add to /workspace/env.sh."
fi

if [ -n "${CONDA_EXE:-}" ]; then
  if [ ! -x "${CONDA_EXE}" ]; then
    err "CONDA_EXE is set but not executable: ${CONDA_EXE}"
  fi
elif ! command -v conda >/dev/null 2>&1; then
  err "conda not found. Install conda in /workspace/conda or set CONDA_EXE."
fi

if ! command -v tmux >/dev/null 2>&1; then
  if [ "${INSTALL}" = "1" ]; then
    apt_install tmux
  else
    err "tmux is missing. Install with: sudo apt-get update && sudo apt-get install -y tmux"
  fi
fi

if ! command -v vim >/dev/null 2>&1; then
  if [ "${INSTALL}" = "1" ]; then
    apt_install vim
  else
    err "vim is missing. Install with: sudo apt-get update && sudo apt-get install -y vim"
  fi
fi

if [ ! -f "${HOME}/.tmux.conf" ]; then
  if [ "${WRITE_TMUX}" = "1" ]; then
    cat > "${HOME}/.tmux.conf" <<'EOF'
set -g mouse on
set -g history-limit 100000
set -g base-index 1
setw -g pane-base-index 1
set -g renumber-windows on
bind r source-file ~/.tmux.conf \; display-message "Reloaded ~/.tmux.conf"
set -g status-interval 5
set -g status-keys vi
setw -g mode-keys vi
EOF
    echo "Wrote ~/.tmux.conf."
  else
    warn "~/.tmux.conf not found. Optional but recommended."
  fi
fi

origin_url="$(git -C "${ROOT}" remote get-url origin 2>/dev/null || true)"
if [[ "${origin_url}" == git@github.com:* ]]; then
  if [ ! -f "${HOME}/.ssh/id_ed25519" ] && [ ! -f "${HOME}/.ssh/id_rsa" ]; then
    if [ "${GENERATE_SSH}" = "1" ]; then
      if [ ! -f "${HOME}/.ssh/id_ed25519" ]; then
        ssh-keygen -t ed25519 -C "runpod" -f "${HOME}/.ssh/id_ed25519" -N ""
      fi
      echo "SSH key generated. Add this to GitHub:"
      cat "${HOME}/.ssh/id_ed25519.pub"
      err "SSH key added locally, but NOT added to GitHub yet."
    else
      err "Git remote uses SSH (git@...), but no SSH key found."
      cat >&2 <<'EOF'
Generate a key and add it to GitHub:
  ssh-keygen -t ed25519 -C "runpod"
  cat ~/.ssh/id_ed25519.pub
Alternative (HTTPS):
  git config --global url."https://github.com/".insteadOf git@github.com:
EOF
    fi
  fi
fi

if [ "${fail}" -ne 0 ]; then
  echo "RunPod bootstrap checks failed. Fix the errors above and rerun." >&2
  exit 1
fi

echo "RunPod bootstrap checks passed."
