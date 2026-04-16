#!/bin/bash
set -euo pipefail

# --- Git config (for agent commits) ---
if [ -n "${GIT_USER_EMAIL:-}" ]; then
  gosu demo git config --global user.email "${GIT_USER_EMAIL}"
fi
if [ -n "${GIT_USER_NAME:-}" ]; then
  gosu demo git config --global user.name "${GIT_USER_NAME}"
fi

# --- Start background services ---
/usr/sbin/sshd 2>/dev/null || echo "Warning: sshd failed to start (SSH access unavailable)"
gosu demo tmux new-session -d -s main 2>/dev/null || true

# --- Launch the main process (drop to demo user) ---
# Everything downstream (bash, mpirun, claude, ...) runs as `demo`, UID 1000.
# MPI jobs therefore execute as non-root — OpenMPI's root-refusal safety check
# is satisfied without resorting to `--allow-run-as-root`.
exec gosu demo "$@"
