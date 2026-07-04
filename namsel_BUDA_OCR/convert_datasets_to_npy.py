#!/usr/bin/env python3
"""Convert Namsel CNN training datasets from pickle (*.pkl) to numpy (*.npy).

Each dataset file is a 2-D array of shape (N, 1025): a label column followed by
1024 pixels (a 32x32 character image). `.npy` is data-only —
`np.load(allow_pickle=False)` cannot execute code, unlike `pickle.load` — and is
smaller than the pickles. Run this once on your datasets directory; afterwards
`dataset.py` loads the `.npy` files.

Usage:
    python convert_datasets_to_npy.py <datasets_dir> [--delete]

`--delete` removes each .pkl only after its .npy has been written and verified
byte-equal.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np


def load_pickle(path):
    """Load a legacy dataset pickle (handles the py2 latin-1 ones)."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin-1")


def main():
    ap = argparse.ArgumentParser(description="Convert dataset *.pkl -> *.npy")
    ap.add_argument("datasets_dir", help="path to namsel_ocr/datasets/")
    ap.add_argument("--delete", action="store_true",
                    help="remove each .pkl after its .npy is verified byte-equal")
    args = ap.parse_args()

    pkls = sorted(glob.glob(os.path.join(args.datasets_dir, "*.pkl")))
    if not pkls:
        sys.exit(f"No *.pkl found in {args.datasets_dir}")

    converted = failed = 0
    for p in pkls:
        npy = p[:-4] + ".npy"
        try:
            arr = np.array(load_pickle(p))
            np.save(npy, arr, allow_pickle=False)
            back = np.load(npy, allow_pickle=False)
            if not (np.array_equal(arr, back) and arr.dtype == back.dtype):
                raise ValueError("round-trip mismatch")
            converted += 1
            print(f"  {os.path.basename(p)} -> {os.path.basename(npy)}  "
                  f"{arr.shape} {arr.dtype}")
            if args.delete:
                os.remove(p)
        except Exception as e:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"  FAILED {os.path.basename(p)}: {e}")

    print(f"\nConverted {converted}, failed {failed}"
          + (" (.pkl deleted)" if args.delete else " (.pkl kept)"))


if __name__ == "__main__":
    main()
