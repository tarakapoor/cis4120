#!/usr/bin/env python3
"""Debug script to inspect imitator.pt structure"""
import torch
from pathlib import Path

# Find imitator.pt
search_paths = [
    Path("./imitator.pt"),
    Path("../imitator.pt"),
    Path("./python/imitator.pt"),
]

model_path = None
for p in search_paths:
    if p.exists():
        model_path = p
        break

if model_path is None:
    print("ERROR: Could not find imitator.pt. Please provide the path.")
    exit(1)

print(f"Loading model from: {model_path}")
print("=" * 60)

# Load the checkpoint
state_dict = torch.load(model_path, map_location='cpu')

print(f"\nType of loaded object: {type(state_dict)}")
print(f"\nKeys in the checkpoint:")

if isinstance(state_dict, dict):
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key:40s} -> Tensor {list(val.shape)}")
        else:
            print(f"  {key:40s} -> {type(val).__name__}")

    # Look for weight keys
    print("\n" + "=" * 60)
    print("2D Weight matrices (what we visualize):")
    weight_keys = [k for k, v in state_dict.items()
                   if isinstance(v, torch.Tensor) and k.endswith(".weight") and v.ndim == 2]

    if weight_keys:
        for k in sorted(weight_keys):
            W = state_dict[k]
            print(f"  {k:40s} -> shape {list(W.shape)}")
    else:
        print("  WARNING: No 2D weight matrices found!")
        print("\n  All keys ending in .weight:")
        for k in sorted([k for k in state_dict.keys() if k.endswith(".weight")]):
            print(f"    {k} -> {state_dict[k].shape}")

    # Check for nested state_dict
    if "state_dict" in state_dict:
        print("\n" + "=" * 60)
        print("Found nested 'state_dict' key! Looking inside:")
        nested = state_dict["state_dict"]
        for key in sorted(nested.keys()):
            val = nested[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key:40s} -> Tensor {list(val.shape)}")

else:
    print(f"Loaded object is not a dict, it's: {type(state_dict)}")
    if hasattr(state_dict, "state_dict"):
        print("But it has a state_dict() method! Calling it...")
        sd = state_dict.state_dict()
        for key in sorted(sd.keys()):
            val = sd[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key:40s} -> Tensor {list(val.shape)}")
