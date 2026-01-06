#!/usr/bin/env python3
"""
Handle SLURM preemption signals to ensure logs are flushed.
"""
import signal
import sys
import time

def handle_preemption(signum, frame):
    """Handle preemption signal by flushing all outputs."""
    print("\n[PREEMPTION] Received signal {}, flushing logs...".format(signum), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(1)  # Give time for logs to write
    print("[PREEMPTION] Logs flushed, exiting gracefully.", flush=True)
    sys.exit(0)

# Register handlers for common SLURM signals
signal.signal(signal.SIGTERM, handle_preemption)  # SLURM preemption
signal.signal(signal.SIGUSR1, handle_preemption)  # SLURM warning

if __name__ == "__main__":
    print("Preemption handler registered", flush=True)
    # Keep the script running
    while True:
        time.sleep(60)
