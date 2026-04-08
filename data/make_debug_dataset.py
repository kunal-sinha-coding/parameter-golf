#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import struct

MAGIC = 20240520
VERSION = 1
HEADER_COUNT = 256
HEADER_BYTES = HEADER_COUNT * 4
TOKEN_BYTES = 2


def truncate_shard(src: pathlib.Path, dst: pathlib.Path, keep_tokens: int) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"missing source shard: {src}")

    with src.open("rb") as f:
        header = list(struct.unpack("<" + "i" * HEADER_COUNT, f.read(HEADER_BYTES)))
        if header[0] != MAGIC or header[1] != VERSION:
            raise ValueError(f"unexpected shard header for {src}")
        num_tokens = int(header[2])
        keep = min(max(keep_tokens, 2), num_tokens)
        payload = f.read(keep * TOKEN_BYTES)

    header[2] = keep
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write(struct.pack("<" + "i" * HEADER_COUNT, *header))
        f.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny debug dataset from existing FineWeb shards")
    parser.add_argument("--source-data-path", required=True)
    parser.add_argument("--debug-data-path", required=True)
    parser.add_argument("--train-tokens", type=int, default=262144)
    parser.add_argument("--val-tokens", type=int, default=131072)
    args = parser.parse_args()

    src_dir = pathlib.Path(args.source_data_path)
    dst_dir = pathlib.Path(args.debug_data_path)
    truncate_shard(src_dir / "fineweb_train_000000.bin", dst_dir / "fineweb_train_000000.bin", args.train_tokens)
    truncate_shard(src_dir / "fineweb_val_000000.bin", dst_dir / "fineweb_val_000000.bin", args.val_tokens)


if __name__ == "__main__":
    main()
