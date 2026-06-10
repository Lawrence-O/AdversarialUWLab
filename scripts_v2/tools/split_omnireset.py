#!/usr/bin/env python3
# Copyright (c) 2025-2026, AdversarialUWLab contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Split the OmniReset reset-state dataset into stratified train/eval files.

Reads the canonical ``resets_<family>.pt`` files under a source directory and
writes per-cell ``resets_<family>_train.pt`` and ``resets_<family>_eval.pt``
into a destination directory, preserving the nested ``initial_state`` dict
structure.  ``partial_assemblies.pt`` is copied unchanged (only used at dataset
generation time).

The split is **stratified per (object_pair, reset_family) cell** using a
deterministic per-cell PRNG seeded by ``hash(pair|family|seed)``.  Each cell
contributes ``eval_fraction`` of its rows to the eval set; the remainder
becomes train.  Train and eval indices are disjoint by construction.

A ``split_manifest.json`` is written at the destination root with:

* the seed / eval-fraction used,
* per-cell row counts (input, train, eval),
* the exact eval indices per cell (so the split can be audited or re-derived
  even if the canonical files change row order).

Defaults match the layout assumed by ``docs/dataset.md``.

Usage::

    python scripts_v2/tools/split_omnireset.py \\
        --src ~/lawony/repos/uwlab-assets/Datasets/OmniReset \\
        --dst ~/lawony/repos/uwlab-assets/Datasets/OmniReset_split_v1 \\
        --eval-fraction 0.10 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from typing import Any

import torch


FAMILIES = (
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
)


def cell_seed(pair: str, family: str, base_seed: int) -> int:
    """Stable 64-bit seed derived from cell identity + base seed."""
    h = hashlib.sha256(f"{pair}|{family}|{base_seed}".encode("utf-8")).digest()
    # take first 8 bytes as unsigned int; PyTorch generator takes int64
    return int.from_bytes(h[:8], "big", signed=False) & ((1 << 63) - 1)


def first_row_count(obj: Any) -> int | None:
    """Find the first ``list`` or 1+D tensor leaf and return its leading length."""
    if isinstance(obj, dict):
        for v in obj.values():
            n = first_row_count(v)
            if n is not None:
                return n
        return None
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, torch.Tensor):
        return obj.shape[0] if obj.ndim >= 1 else None
    return None


def slice_in_place(obj: Any, indices: torch.Tensor) -> Any:
    """Return a new nested structure with every per-row leaf sliced by ``indices``.

    ``obj`` may contain dicts, lists (treated as per-row), or tensors. Non-row
    scalars / unrelated tensors pass through unchanged.
    """
    if isinstance(obj, dict):
        return {k: slice_in_place(v, indices) for k, v in obj.items()}
    if isinstance(obj, list):
        # We assume top-level lists are per-row arrays. Index-select.
        idx_list = indices.tolist()
        return [obj[i] for i in idx_list]
    if isinstance(obj, torch.Tensor):
        if obj.ndim >= 1 and obj.shape[0] == _row_count_hint.get("n"):
            return obj.index_select(0, indices)
        return obj
    return obj


# Module-level hint so slice_in_place can compare a tensor's leading dim against
# the expected N for the cell currently being processed (avoids slicing the wrong
# axis of unrelated tensors).
_row_count_hint: dict[str, int] = {}


def split_cell(src_path: str, dst_dir: str, family: str, base_seed: int, pair: str, eval_fraction: float) -> dict:
    data = torch.load(src_path, map_location="cpu", weights_only=False)
    payload = data.get("initial_state", data)
    n = first_row_count(payload)
    if n is None:
        raise RuntimeError(f"could not determine row count for {src_path}")

    _row_count_hint["n"] = n

    seed = cell_seed(pair, family, base_seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_eval = int(round(n * eval_fraction))
    eval_idx = perm[:n_eval].sort().values  # sorted for nicer audit, semantically same set
    train_idx = perm[n_eval:].sort().values

    def write(name: str, idx: torch.Tensor) -> str:
        sliced = slice_in_place(payload, idx)
        out = {"initial_state": sliced} if "initial_state" in data else sliced
        out_path = os.path.join(dst_dir, f"resets_{family}_{name}.pt")
        torch.save(out, out_path)
        return out_path

    train_path = write("train", train_idx)
    eval_path = write("eval", eval_idx)

    return {
        "n_total": n,
        "n_train": int(train_idx.numel()),
        "n_eval": int(eval_idx.numel()),
        "eval_indices": eval_idx.tolist(),
        "seed": seed,
        "train_path": os.path.relpath(train_path, dst_dir),
        "eval_path": os.path.relpath(eval_path, dst_dir),
    }


def main() -> int:
    here = os.path.expanduser("~/lawony/repos/uwlab-assets/Datasets")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(here, "OmniReset"),
                        help="Source dataset root (containing Resets/<pair>/...).")
    parser.add_argument("--dst", default=os.path.join(here, "OmniReset_split_v1"),
                        help="Destination split-dataset root.")
    parser.add_argument("--eval-fraction", type=float, default=0.10,
                        help="Per-cell eval fraction (default 0.10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed combined with (pair, family) for the per-cell PRNG.")
    parser.add_argument("--copy-grasps", action="store_true", default=True,
                        help="Symlink Grasps/ from src into dst (not split).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove dst if it already exists.")
    args = parser.parse_args()

    src_resets = os.path.join(args.src, "Resets")
    dst_resets = os.path.join(args.dst, "Resets")

    if not os.path.isdir(src_resets):
        print(f"ERROR: --src/Resets not found at {src_resets}", file=sys.stderr)
        return 2

    if os.path.exists(args.dst):
        if not args.overwrite:
            print(f"ERROR: {args.dst} already exists; pass --overwrite to replace.", file=sys.stderr)
            return 2
        shutil.rmtree(args.dst)

    os.makedirs(dst_resets, exist_ok=True)

    manifest: dict[str, Any] = {
        "src": os.path.abspath(args.src),
        "dst": os.path.abspath(args.dst),
        "eval_fraction": args.eval_fraction,
        "base_seed": args.seed,
        "families": list(FAMILIES),
        "cells": {},
    }

    pairs = sorted(d for d in os.listdir(src_resets) if os.path.isdir(os.path.join(src_resets, d)))
    print(f"Found {len(pairs)} object pairs under {src_resets}")
    for pair in pairs:
        src_pair = os.path.join(src_resets, pair)
        dst_pair = os.path.join(dst_resets, pair)
        os.makedirs(dst_pair, exist_ok=True)
        print(f"\n[{pair}]")
        for family in FAMILIES:
            src_path = os.path.join(src_pair, f"resets_{family}.pt")
            if not os.path.isfile(src_path):
                print(f"  skip {family}: missing source file")
                continue
            info = split_cell(src_path, dst_pair, family, args.seed, pair, args.eval_fraction)
            manifest["cells"][f"{pair}/{family}"] = {k: v for k, v in info.items() if k != "eval_indices"}
            # eval indices kept separately so the top-level manifest stays compact
            manifest["cells"][f"{pair}/{family}"]["eval_indices_file"] = f"manifest_eval_indices/{pair}__{family}.json"
            idx_dir = os.path.join(args.dst, "manifest_eval_indices")
            os.makedirs(idx_dir, exist_ok=True)
            with open(os.path.join(idx_dir, f"{pair}__{family}.json"), "w") as f:
                json.dump(info["eval_indices"], f)
            print(f"  {family:<37} N={info['n_total']:>6}  train={info['n_train']:>6}  eval={info['n_eval']:>6}")
        # partial_assemblies.pt: copy unchanged (not consumed as resets at train time)
        pa_src = os.path.join(src_pair, "partial_assemblies.pt")
        if os.path.isfile(pa_src):
            shutil.copy2(pa_src, os.path.join(dst_pair, "partial_assemblies.pt"))

    if args.copy_grasps:
        grasps_src = os.path.join(args.src, "Grasps")
        if os.path.isdir(grasps_src):
            grasps_dst = os.path.join(args.dst, "Grasps")
            if not os.path.exists(grasps_dst):
                os.symlink(grasps_src, grasps_dst)
            print(f"\nGrasps symlinked: {grasps_dst} -> {grasps_src}")

    manifest_path = os.path.join(args.dst, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")

    # Quick summary
    total_n = sum(c["n_total"] for c in manifest["cells"].values())
    total_train = sum(c["n_train"] for c in manifest["cells"].values())
    total_eval = sum(c["n_eval"] for c in manifest["cells"].values())
    print(f"\nGrand totals: N={total_n:,}  train={total_train:,}  eval={total_eval:,}  ({100*total_eval/total_n:.2f}% eval)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
