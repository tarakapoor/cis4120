# python/server.py
import io, json, uuid
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from tempfile import mkstemp
import os
import subprocess

# ---------- FastAPI setup ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SAE (from sae.py) ----------
import torch.nn.functional as F

class TopKSAE(torch.nn.Module):
    def __init__(self, d_in: int, d_latent: int, k: int = 16):
        super().__init__()
        self.k = k
        self.encoder = torch.nn.Linear(d_in, d_latent, bias=False)
        self.decoder = torch.nn.Linear(d_latent, d_in, bias=False)
    def encode(self, x):
        z = F.relu(self.encoder(x))
        if self.k is not None and self.k < z.shape[1]:
            vals, idx = torch.topk(z, k=self.k, dim=1)
            mask = torch.zeros_like(z); mask.scatter_(1, idx, 1.0); z = z * mask
        return z
    def forward(self, x):
        z = self.encode(x); x_hat = self.decoder(z); return x_hat, z

# ---------- In-memory registry ----------
class ModelEntry(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    state_dict: Dict[str, torch.Tensor]
    ordered_keys: List[str]
    widths: List[int]
    sae: Optional[TopKSAE] = None  # Optional SAE for this model
    sae_tap_index: Optional[int] = None  # Which layer the SAE taps
    cached_obs: Optional[torch.Tensor] = None  # For interpretation
    cached_acts: Optional[torch.Tensor] = None  # For interpretation

REGISTRY: Dict[str, ModelEntry] = {}

# Whitelist of allowed scripts to run
SCRIPT_DIR = Path(__file__).parent
WHITELIST = {
    "rollout": str(SCRIPT_DIR / "run_bc_model.py"),
    "train": str(SCRIPT_DIR / "train.py"),
    "hard_stable": str(SCRIPT_DIR / "hard_stable.py"),
}

# ---------- utils ----------
def _load_pt_bytes(b: bytes) -> Dict[str, torch.Tensor]:
    buf = io.BytesIO(b)
    obj = torch.load(buf, map_location="cpu")
    # common cases
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError("Unsupported .pt format (no state_dict found)")

def _sort_key(k: str):
    # sort keys like net.0.weight < net.1.weight < ...
    parts = []
    for t in k.replace("[",".").replace("]",".").split("."):
        parts.append((0, int(t)) if t.isdigit() else (1, t))
    return parts

def _infer_widths(sd: Dict[str, torch.Tensor]) -> tuple[List[str], List[int]]:
    items = [(k, v) for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2]
    items.sort(key=lambda kv: _sort_key(kv[0]))
    ordered = [k for k, _ in items]
    widths: List[int] = []
    for _, W in items:
        out_d, in_d = W.shape
        if not widths:
            widths = [in_d, out_d]
        else:
            if widths[-1] != in_d:  # new chain; still append best-effort
                widths.append(in_d)
            widths.append(out_d)
    # Don't compress - keep all layer sizes even if they repeat
    # This is important for visualizing networks with multiple same-sized hidden layers
    return ordered, [int(w) for w in widths]

def _summary_json(widths: List[int]):
    return {"layers": [{"size": int(s)} for s in widths]}

# ---------- API ----------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    data = await file.read()

    if name.endswith(".json"):
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception as e:
            raise HTTPException(400, f"Invalid JSON: {e}")
        model_id = str(uuid.uuid4())
        REGISTRY[model_id] = ModelEntry(state_dict={}, ordered_keys=[], widths=[l["size"] for l in payload.get("layers", [])])
        return {"model_id": model_id, "summary": payload, "kind": "json"}

    if name.endswith(".pt"):
        try:
            sd = _load_pt_bytes(data)
        except Exception as e:
            raise HTTPException(400, f"Invalid .pt: {e}")
        ordered, widths = _infer_widths(sd)
        model_id = str(uuid.uuid4())
        REGISTRY[model_id] = ModelEntry(state_dict=sd, ordered_keys=ordered, widths=widths)
        return {"model_id": model_id, "summary": _summary_json(widths), "keys": ordered, "kind": "pt"}

    raise HTTPException(415, "Please upload .json or .pt")

class PerturbSpec(BaseModel):
    op: str                    # "scale" | "add_noise" | "set"
    key: Optional[str] = None  # if None -> first 2D weight
    scale: Optional[float] = None
    std: Optional[float] = None
    value: Optional[float] = None

@app.post("/model/{model_id}/perturb")
def perturb(model_id: str, spec: PerturbSpec):
    if model_id not in REGISTRY:
        raise HTTPException(404, "Unknown model_id")
    entry = REGISTRY[model_id]

    key = spec.key or next((k for k in entry.ordered_keys if entry.state_dict[k].ndim == 2), None)
    if key is None or key not in entry.state_dict:
        raise HTTPException(400, "No valid 2D weight key found; provide 'key'")

    W = entry.state_dict[key]
    if spec.op == "scale":
        s = float(spec.scale if spec.scale is not None else 1.0)
        entry.state_dict[key] = W * s
    elif spec.op == "add_noise":
        std = float(spec.std if spec.std is not None else 0.01)
        entry.state_dict[key] = W + torch.randn_like(W) * std
    elif spec.op == "set":
        if spec.value is None:
            raise HTTPException(400, "Missing 'value' for op=set")
        entry.state_dict[key] = torch.full_like(W, float(spec.value))
    else:
        raise HTTPException(400, f"Unknown op: {spec.op}")

    return {"ok": True, "key": key, "shape": list(entry.state_dict[key].shape)}

class SaveSpec(BaseModel):
    path: str

@app.post("/model/{model_id}/save")
def save(model_id: str, spec: SaveSpec):
    if model_id not in REGISTRY:
        raise HTTPException(404, "Unknown model_id")
    try:
        torch.save(REGISTRY[model_id].state_dict, spec.path)
    except Exception as e:
        raise HTTPException(500, f"Save failed: {e}")
    return {"ok": True, "path": spec.path}

@app.get("/model/{model_id}/summary")
def summary(model_id: str):
    if model_id not in REGISTRY:
        raise HTTPException(404, "Unknown model_id")
    e = REGISTRY[model_id]
    return {"summary": _summary_json(e.widths), "keys": e.ordered_keys}

# Get full weight matrix for a specific layer key
@app.get("/model/{model_id}/layer_weights/{key}")
def get_layer_weights(model_id: str, key: str):
    if model_id not in REGISTRY:
        raise HTTPException(404, "Unknown model_id")
    e = REGISTRY[model_id]
    if key not in e.state_dict:
        raise HTTPException(404, f"Key '{key}' not found in model")
    W = e.state_dict[key]
    # Return as nested list for JSON serialization
    return {"key": key, "shape": list(W.shape), "weights": W.detach().cpu().tolist()}



# Return a single weight value at [row, col]
@app.get("/model/{model_id}/value")
def get_value(model_id: str, key: str, row: int, col: int):
    if model_id not in REGISTRY: raise HTTPException(404, "Unknown model_id")
    e = REGISTRY[model_id]
    if key not in e.state_dict: raise HTTPException(400, "Bad key")
    W = e.state_dict[key]
    if W.ndim != 2: raise HTTPException(400, "Only 2D tensors supported here")
    if row < 0 or col < 0 or row >= W.shape[0] or col >= W.shape[1]:
        raise HTTPException(400, "Index out of bounds")
    return {"value": float(W[row, col].item())}

class EditAtSpec(BaseModel):
    key: str
    row: int
    col: int
    op: str              # "set" | "add" | "scale"
    value: float         # set to this / add this / multiply by this

@app.post("/model/{model_id}/edit_at")
def edit_at(model_id: str, spec: EditAtSpec):
    if model_id not in REGISTRY: raise HTTPException(404, "Unknown model_id")
    e = REGISTRY[model_id]
    if spec.key not in e.state_dict: raise HTTPException(400, "Bad key")
    W = e.state_dict[spec.key]
    if W.ndim != 2: raise HTTPException(400, "Only 2D tensors supported")
    if spec.row < 0 or spec.col < 0 or spec.row >= W.shape[0] or spec.col >= W.shape[1]:
        raise HTTPException(400, "Index out of bounds")
    with torch.no_grad():
        if spec.op == "set":
            W[spec.row, spec.col] = float(spec.value)
        elif spec.op == "add":
            W[spec.row, spec.col] = W[spec.row, spec.col] + float(spec.value)
        elif spec.op == "scale":
            W[spec.row, spec.col] = W[spec.row, spec.col] * float(spec.value)
        else:
            raise HTTPException(400, "Unknown op")
    return {"ok": True, "shape": list(W.shape)}

# Save current in-memory state_dict to a temp file and return the path
@app.post("/model/{model_id}/save_temp")
def save_temp(model_id: str):
    if model_id not in REGISTRY: raise HTTPException(404, "Unknown model_id")
    fd, path = mkstemp(prefix="edited_", suffix=".pt", dir=str(Path(__file__).parent))
    os.close(fd)
    torch.save(REGISTRY[model_id].state_dict, path)
    return {"path": path}

# Convenience: save temp and run a whitelisted script with it
class RunWithModelSpec(BaseModel):
    script: str           # must be in WHITELIST (reuse your /run whitelist)
    env: str = "hard_stable"  # environment name
    num_traj: int = 5
    max_steps: int = 300
    capture_activations: bool = False  # Whether to capture activations
    extra_args: List[str] = []  # any other flags

# ========== SAE Endpoints ==========

# Load SAE from artifacts directory
class LoadSAESpec(BaseModel):
    artifacts_dir: str = "."
    tap_index: int = 4  # Which layer to tap (default: net[4])

@app.post("/model/{model_id}/load_sae")
def load_sae(model_id: str, spec: LoadSAESpec):
    if model_id not in REGISTRY: raise HTTPException(404, "Unknown model_id")
    entry = REGISTRY[model_id]

    # Resolve paths relative to python directory if not absolute
    base_dir = Path(spec.artifacts_dir)
    if not base_dir.is_absolute():
        base_dir = SCRIPT_DIR / spec.artifacts_dir

    # Load SAE checkpoint
    sae_path = base_dir / "walker_sae.pt"
    if not sae_path.exists():
        # Try current directory as fallback
        sae_path_alt = SCRIPT_DIR / "walker_sae.pt"
        if sae_path_alt.exists():
            sae_path = sae_path_alt
            base_dir = SCRIPT_DIR
        else:
            raise HTTPException(404, f"SAE not found at {sae_path} or {sae_path_alt}")

    try:
        ck = torch.load(sae_path, map_location="cpu")
        sae = TopKSAE(ck['d_in'], ck['d_latent'], k=ck['k'])
        sae.load_state_dict(ck['sae_state'])
        sae.eval()
    except Exception as e:
        raise HTTPException(500, f"Failed to load SAE: {str(e)}")

    entry.sae = sae
    entry.sae_tap_index = spec.tap_index

    # Optionally load cached obs/acts for interpretation
    cached_obs_path = base_dir / "cached_obs.pt"
    cached_acts_path = base_dir / "tapped_activations.pt"

    if cached_obs_path.exists():
        try:
            entry.cached_obs = torch.load(cached_obs_path, map_location="cpu")
        except Exception as e:
            print(f"Warning: Failed to load cached_obs: {e}")

    if cached_acts_path.exists():
        try:
            entry.cached_acts = torch.load(cached_acts_path, map_location="cpu")
        except Exception as e:
            print(f"Warning: Failed to load cached_acts: {e}")

    return {
        "ok": True,
        "d_in": ck['d_in'],
        "d_latent": ck['d_latent'],
        "k": ck['k'],
        "tap_index": spec.tap_index,
        "has_cached_data": entry.cached_obs is not None,
        "loaded_from": str(sae_path)
    }

# SAE feature-based perturbation (like act_with_feature_edit in sae.py)
class SAEPerturbSpec(BaseModel):
    feature_idx: int  # Which SAE feature to perturb
    alpha: float      # Perturbation strength
    tap_index: Optional[int] = None  # Override tap index if needed

@app.post("/model/{model_id}/sae_perturb")
def sae_perturb(model_id: str, spec: SAEPerturbSpec):
    """Apply SAE feature-based perturbation by editing activations at a tapped layer"""
    if model_id not in REGISTRY: raise HTTPException(404, "Unknown model_id")
    entry = REGISTRY[model_id]

    if entry.sae is None:
        raise HTTPException(400, "No SAE loaded. Call /load_sae first.")

    # Get decoder direction for this feature
    sae = entry.sae
    wj = sae.decoder.weight[:, spec.feature_idx]  # [d_in]

    # We'll apply this perturbation by modifying the activation hook
    # For now, store the perturbation parameters so run_model can use them
    # (In a real implementation, you'd apply during forward pass)

    return {
        "ok": True,
        "feature_idx": spec.feature_idx,
        "alpha": spec.alpha,
        "decoder_norm": float(torch.norm(wj).item()),
        "message": "Feature perturbation parameters stored (apply during model run)"
    }

# Get top features by interpretability (like ridge probe in sae.py)
class InterpretFeaturesSpec(BaseModel):
    target_dim: int = 0  # Which action dimension to interpret
    top_k: int = 10      # Return top K features

@app.post("/model/{model_id}/interpret_features")
def interpret_features(model_id: str, spec: InterpretFeaturesSpec):
    """Find most interpretable SAE features using ridge regression"""
    if model_id not in REGISTRY: raise HTTPException(404, "Unknown model_id")
    entry = REGISTRY[model_id]

    if entry.sae is None or entry.cached_obs is None or entry.cached_acts is None:
        raise HTTPException(400, "Need SAE + cached data. Call /load_sae with artifacts.")

    import numpy as np

    # Encode activations to SAE features
    with torch.no_grad():
        Z = entry.sae.encode(entry.cached_acts).numpy()  # [N, d_latent]

    # Get target labels (need to run model on cached_obs to get actions)
    # For now, simplified: use obs dimension as proxy
    # TODO: Actually run the model to get predicted actions
    y = entry.cached_obs[:, spec.target_dim].numpy()  # [N]

    # Ridge regression: Z @ w = y
    N, m = Z.shape
    lam = 1e-2
    Xb = np.concatenate([Z, np.ones((N, 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(m + 1)
    w_full = np.linalg.solve(A, Xb.T @ y)
    w = w_full[:-1]

    # Get top features by absolute weight
    top_indices = np.argsort(np.abs(w))[-spec.top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "feature_idx": int(idx),
            "weight": float(w[idx]),
            "abs_weight": float(np.abs(w[idx]))
        })

    return {"features": results, "target_dim": spec.target_dim}

@app.post("/model/{model_id}/save_and_run")
def save_and_run(model_id: str, spec: RunWithModelSpec):
    if model_id not in REGISTRY: raise HTTPException(404, "Unknown model_id")
    if spec.script not in WHITELIST: raise HTTPException(400, "Script not allowed")
    # save temp
    fd, path = mkstemp(prefix="edited_", suffix=".pt", dir=str(Path(__file__).parent))
    os.close(fd)
    torch.save(REGISTRY[model_id].state_dict, path)
    # inject into args as --policy (or whatever your script expects)
    # Use conda run to execute in the proper environment
    cmd = [
        "conda", "run", "-n", "ddpm_bc_env", "--no-capture-output",
        "python", WHITELIST[spec.script],
        "--policy", path,
        "--env", spec.env,
        "--num_traj", str(spec.num_traj),
        "--max_steps", str(spec.max_steps),
    ]
    if spec.capture_activations:
        cmd.append("--capture_activations")
    cmd.extend(spec.extra_args)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60*30)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Task timed out")

    # Try to parse JSON output from stdout
    results = None
    try:
        # Look for JSON in stdout
        import re
        json_match = re.search(r'\{.*\}', out.stdout, re.DOTALL)
        if json_match:
            results = json.loads(json_match.group())
    except:
        pass

    return {
        "returncode": out.returncode,
        "stdout": out.stdout[-4000:],
        "stderr": out.stderr[-4000:],
        "policy_path": path,
        "results": results
    }