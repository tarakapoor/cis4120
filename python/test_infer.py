#!/usr/bin/env python3
"""Test the _infer_widths function"""
import torch
from typing import Dict, List

def _sort_key(k: str):
    parts = []
    for t in k.replace("[",".").replace("]",".").split("."):
        parts.append((0, int(t)) if t.isdigit() else (1, t))
    return parts

def _infer_widths(sd: Dict[str, torch.Tensor]) -> tuple[List[str], List[int]]:
    items = [(k, v) for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2]
    items.sort(key=lambda kv: _sort_key(kv[0]))
    ordered = [k for k, _ in items]

    print(f"Found {len(items)} weight matrices:")
    for k, W in items:
        print(f"  {k}: {list(W.shape)}")

    widths: List[int] = []
    for idx, (k, W) in enumerate(items):
        out_d, in_d = W.shape
        print(f"\n[{idx}] Processing {k}: in_d={in_d}, out_d={out_d}")
        if not widths:
            widths = [in_d, out_d]
            print(f"    First layer! widths = {widths}")
        else:
            print(f"    Current widths = {widths}, widths[-1] = {widths[-1]}")
            if widths[-1] != in_d:
                print(f"    Mismatch! {widths[-1]} != {in_d}, appending {in_d}")
                widths.append(in_d)
            widths.append(out_d)
            print(f"    After appending out_d: widths = {widths}")

    # Don't compress - keep all layer sizes even if they repeat
    print(f"\nFinal widths (no compression): {widths}")

    return ordered, [int(w) for w in widths]

# Load and test
sd = torch.load("imitator.pt", map_location='cpu')
ordered, widths = _infer_widths(sd)

print("\n" + "=" * 60)
print(f"Ordered keys: {ordered}")
print(f"Widths: {widths}")
print(f"Number of layers in visualization: {len(widths)}")
