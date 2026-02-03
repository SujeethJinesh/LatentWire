#!/usr/bin/env python3
import argparse
import numpy as np

from quantization.kvwire.kvwire_v1 import KVWireConfig, pack, unpack


def main():
    parser = argparse.ArgumentParser(description="Tiny KVWire roundtrip equivalence test.")
    parser.add_argument("--quant", choices=["int8", "int4"], default="int8")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    k = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    indices = np.arange(k.shape[2], dtype=np.uint16)
    cfg = KVWireConfig(wire_quant_mode=args.quant)

    blob = pack({"k": k, "v": v, "indices": indices}, cfg)
    out = unpack(blob, cfg)
    max_err = float(np.max(np.abs(out["k"] - k)))
    print(f"KVWire {args.quant} roundtrip ok; max_err={max_err:.6f}, bytes={len(blob)}")


if __name__ == "__main__":
    main()
