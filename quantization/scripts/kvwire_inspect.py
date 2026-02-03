#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from quantization.kvwire.kvwire_v1 import unpack, KVWireConfig


def main():
    parser = argparse.ArgumentParser(description="Inspect KVWire blob metadata and section sizes.")
    parser.add_argument("blob", help="Path to KVWire blob")
    args = parser.parse_args()

    blob_path = Path(args.blob)
    blob = blob_path.read_bytes()
    out = unpack(blob, KVWireConfig())
    meta = out.get("meta", {})
    sections = meta.get("sections", {})
    print(json.dumps({
        "blob": str(blob_path),
        "total_bytes": len(blob),
        "meta": meta,
        "sections": sections,
    }, indent=2))


if __name__ == "__main__":
    main()
