# bc_sae_walker_demo.py
# End-to-end (RESUME-ABLE): BC -> SAE -> pick feature -> side-by-side interventions video
# With SHARP demo tweaks: alpha calibration + feature-activity gating + clean writer/env cleanup.

import os
# Headless Mujoco (try EGL first; fallback to OSMesa by uncommenting next line if needed)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# os.environ.setdefault("MUJOCO_GL", "osmesa")

import json
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from tqdm import tqdm

# ---- Your helpers (must exist) ----
from rollout import TrajLoader, CollectTrajs


# =========================
# 1) BC policy nets
# =========================
class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_len=1, hidden_size=256):
        super().__init__()
        output_dim = act_dim * chunk_len
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.9, eps=1e-5),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.9, eps=1e-5),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim)
        )
    def forward(self, x): return self.net(x)

class MLP_no_batch_bias(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_len=1, hidden_size=256):
        super().__init__()
        output_dim = act_dim * chunk_len
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim, bias=False)
        )
    def forward(self, x): return self.net(x)

def train_bc(model, loader, device, lr, epochs):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    model.train()
    losses = []
    for ep in tqdm(range(epochs), desc="BC training"):
        total = 0.0
        for obs, acts in loader:
            obs = obs.to(device); acts = acts.to(device)
            pred = model(obs)
            loss = crit(pred, acts.flatten(start_dim=1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item() * obs.size(0)
        sch.step()
        losses.append(total / len(loader.dataset))
    print(f"[BC] Final loss: {losses[-1]:.6f}")
    return model, losses


# =========================
# 2) Activation tap
# =========================
class ActivationTap:
    def __init__(self, module):
        self.buffer = []; self.h = module.register_forward_hook(self._hook)
    def _hook(self, m, i, o): self.buffer.append(o.detach().float().cpu())
    def stacked(self): return torch.cat(self.buffer, dim=0) if self.buffer else torch.empty(0)
    def remove(self): self.h.remove()

@torch.no_grad()
def collect_activations(model, loader, device, tap_module, collect_max_batches: Optional[int]=None):
    model.eval()
    tap = ActivationTap(tap_module)
    n = 0
    for obs, _ in tqdm(loader, desc="Collecting activations"):
        _ = model(obs.to(device)); n += 1
        if collect_max_batches and n >= collect_max_batches: break
    X = tap.stacked(); tap.remove()
    return X


# =========================
# 3) Sparse Autoencoder (Top-K)
# =========================
class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_latent: int, k: int = 16):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_in, d_latent, bias=False)
        self.decoder = nn.Linear(d_latent, d_in, bias=False)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
    def encode(self, x):
        z = F.relu(self.encoder(x))
        if self.k is not None and self.k < z.shape[1]:
            vals, idx = torch.topk(z, k=self.k, dim=1)
            mask = torch.zeros_like(z); mask.scatter_(1, idx, 1.0); z = z * mask
        return z
    def forward(self, x):
        z = self.encode(x); x_hat = self.decoder(z); return x_hat, z

def sae_loss(x, x_hat): return F.mse_loss(x_hat, x)

def train_sae(acts: torch.Tensor, d_latent: int, k: int, steps: int, batch_size: int, lr: float, device: str):
    ds = torch.utils.data.TensorDataset(acts)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    sae = TopKSAE(acts.shape[1], d_latent, k=k).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    sae.train()
    it = iter(dl); pbar = tqdm(range(steps), desc="SAE training")
    for step in pbar:
        try: (xb,) = next(it)
        except StopIteration: it = iter(dl); (xb,) = next(it)
        xb = xb.to(device); xh, _ = sae(xb); loss = sae_loss(xb, xh)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if (step+1) % 500 == 0: pbar.set_postfix(loss=float(loss.item()))
    sae.eval(); return sae

@torch.no_grad()
def encode_Z(sae, X, device):
    # Safe & simple: use the model's own Top-K path
    sae.eval()
    return sae.encode(X.to(device)).cpu()


# =========================
# Helpers: load & infer dims from BC ckpt
# =========================
def load_state(path, map_location):
    return torch.load(path, map_location=map_location)

def infer_dims_from_bc_ckpt(state_dict, chunk_len=1):
    # Try BN layout: net.0.weight (Linear), net.8.weight (final Linear)
    if "net.0.weight" in state_dict and "net.8.weight" in state_dict:
        first = state_dict["net.0.weight"]  # [hidden, obs_dim]
        last  = state_dict["net.8.weight"]  # [out_dim, hidden]
        hidden_size = first.shape[0]
        obs_dim     = first.shape[1]
        out_dim     = last.shape[0]
        act_dim     = out_dim // chunk_len
        uses_bn     = True
        return obs_dim, hidden_size, act_dim, uses_bn
    # Try alternative last index (e.g., biasless variant)
    for last_idx in [6, 8]:
        k0 = "net.0.weight"; kL = f"net.{last_idx}.weight"
        if k0 in state_dict and kL in state_dict:
            first = state_dict[k0]; last = state_dict[kL]
            hidden_size = first.shape[0]
            obs_dim     = first.shape[1]
            out_dim     = last.shape[0]
            act_dim     = out_dim // chunk_len
            uses_bn     = (last_idx == 8)
            return obs_dim, hidden_size, act_dim, uses_bn
    raise RuntimeError("Could not infer dims from BC checkpoint; unexpected layer keys.")


# =========================
# 4) Interventions (override)
# =========================
class OverrideOutput:
    def __init__(self, module, provider):
        self.provider = provider; self.h = module.register_forward_hook(self._hook)
    def _hook(self, m, i, o): return self.provider()
    def remove(self): self.h.remove()

@torch.no_grad()
def act_with_feature_edit(model, tap_module, sae, obs_batch, wj, alpha, device):
    """Two-pass: capture baseline tap -> override with edited -> forward"""
    obs_t = obs_batch.to(device)
    tap_cache = {}
    def capture(m, i, o): tap_cache['x'] = o.detach()
    h = tap_module.register_forward_hook(lambda m,i,o: capture(m,i,o))
    base = model(obs_t); h.remove()
    X = tap_cache['x']               # [B, D]
    edit = X + alpha * wj.unsqueeze(0)
    override = OverrideOutput(tap_module, lambda: edit)
    out = model(obs_t); override.remove()
    return base, out


# =========================
# 5) Alpha calibration + feature gating (NEW)
# =========================
@torch.no_grad()
def calibrate_alpha(model, tap_module, sae, ob_np, wj, device, target_dim, target_delta=0.35, eps=0.05, max_alpha=3.0):
    """Estimate local gain d a_k / d alpha and pick alpha so unclipped delta ~= target_delta."""
    ob_t = torch.from_numpy(ob_np).float().unsqueeze(0)
    base, _ = act_with_feature_edit(model, tap_module, sae, ob_t, wj, 0.0, device)
    a0 = base.squeeze(0).cpu().numpy()
    _, a_pos = act_with_feature_edit(model, tap_module, sae, ob_t, wj, eps, device)
    a1 = a_pos.squeeze(0).cpu().numpy()
    gain = (a1[target_dim] - a0[target_dim]) / max(eps, 1e-6)
    if abs(gain) < 1e-6:
        print("[Calibrate] Tiny gain; falling back to provided --alpha.")
        return None
    alpha = np.clip(target_delta / gain, -max_alpha, +max_alpha)
    return float(abs(alpha))  # symmetric +/- usage

@torch.no_grad()
def feature_activity(model, tap_module, sae, ob_np, wj, device):
    """Score current feature activity along decoder direction d_j (dot with tap)."""
    ob_t = torch.from_numpy(ob_np).float().unsqueeze(0).to(device)
    tap_cache = {}
    def capture(m,i,o): tap_cache['x']=o.detach()
    h = tap_module.register_forward_hook(lambda m,i,o: capture(m,i,o))
    _ = model(ob_t); h.remove()
    X = tap_cache['x']  # [1, D]
    return float((X @ wj).squeeze().detach().cpu().numpy())


# =========================
# 6) Main (with --resume / --video_only)
# =========================
def main():
    ap = argparse.ArgumentParser("BC + SAE + Joint-Torque Feature + Video (resume-able, calibrated)")
    # Data/BC
    ap.add_argument('--env', type=str, required=True)
    ap.add_argument('--policy', type=str, required=False, help='expert ckpt path (only needed if collecting data)')
    ap.add_argument('--chunk_len', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--noise_scale', type=float, default=0.0)
    ap.add_argument('--prop_noised', type=float, default=0.0)
    ap.add_argument('--max_timesteps', type=int, default=300)
    ap.add_argument('--num_traj', type=int, default=200)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--hidden_size', type=int, default=256)
    ap.add_argument('--use_bn_model', action='store_true')
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-3)

    # SAE
    ap.add_argument('--tap_index', type=int, default=4)
    ap.add_argument('--sae_latent_mult', type=int, default=4)
    ap.add_argument('--sae_topk', type=int, default=16)
    ap.add_argument('--sae_steps', type=int, default=50000)
    ap.add_argument('--sae_batch', type=int, default=1024)
    ap.add_argument('--sae_lr', type=float, default=1e-3)
    ap.add_argument('--collect_max_batches', type=int, default=None)

    # Demo selection & video
    ap.add_argument('--target_act_idx', type=int, default=0)
    ap.add_argument('--alpha', type=float, default=2.0, help='fallback alpha if calibration fails')
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--outfile', type=str, default='walker_intervention.mp4')

    # Resume controls
    ap.add_argument('--resume', action='store_true', help='Load existing artifacts and skip training/collection where possible')
    ap.add_argument('--video_only', action='store_true', help='Assume selection done; only render the video')
    ap.add_argument('--artifacts_dir', type=str, default='.', help='Directory to read/write artifacts')

    args = ap.parse_args()
    device = args.device
    A = lambda name: os.path.join(args.artifacts_dir, name)

    # ---------- Artifact paths ----------
    acts_path = A('tapped_activations.pt')
    cached_obs_path = A('cached_obs.pt')
    bc_path = A('imitator.pt')
    sae_path = A('walker_sae.pt')
    sel_path = A('selected_feature.json')

    # ---------- Helper: loader only if needed ----------
    def build_loader():
        if args.policy is None:
            raise RuntimeError("build_loader() called but --policy not provided.")
        trajs = CollectTrajs(
            env_name=args.env, policy_path=args.policy,
            max_timesteps=args.max_timesteps, num_trajectories=args.num_traj,
            noise_scale=args.noise_scale, prop_noised=args.prop_noised,
            deterministic=True, seed=42
        )
        return TrajLoader(trajs, chunk_len=args.chunk_len, batch_size=args.batch_size, device='cpu')

    # ---------- Build/Load BC model (resume-friendly) ----------
    obs_dim = act_dim = None
    inferred_hidden = args.hidden_size

    if os.path.exists(cached_obs_path):
        cached_obs = torch.load(cached_obs_path, map_location='cpu')
        obs_dim = int(cached_obs.shape[1])
    if os.path.exists(bc_path):
        bc_sd = load_state(bc_path, map_location='cpu')
        try:
            inf_obs_dim, inf_hidden, inf_act_dim, _ = infer_dims_from_bc_ckpt(bc_sd, chunk_len=args.chunk_len)
            if obs_dim is None: obs_dim = inf_obs_dim
            act_dim = inf_act_dim
            inferred_hidden = inf_hidden
        except Exception:
            pass

    if obs_dim is None and not args.resume:
        loader = build_loader()
        sample_obs, sample_act = next(iter(loader))
        obs_dim = int(sample_obs.shape[1])
        act_dim = int(sample_act.shape[1] if sample_act.ndim == 2 else sample_act.shape[2])

    if obs_dim is None and args.resume and os.path.exists(bc_path):
        bc_sd = load_state(bc_path, map_location='cpu')
        for k, W in bc_sd.items():
            if isinstance(W, torch.Tensor) and W.ndim == 2:
                obs_dim = int(W.shape[1]); inferred_hidden = int(W.shape[0]); break
    if obs_dim is None and args.resume:
        raise AssertionError("Could not infer obs_dim from artifacts. Provide --policy once to create cached_obs.pt.")

    ckpt_says_bn = None
    if os.path.exists(bc_path):
        bc_sd = load_state(bc_path, map_location='cpu')
        has_bn_keys = any(k.endswith("running_mean") or k.endswith("running_var") for k in bc_sd.keys())
        has_bn_layers = any(k.startswith("net.1.") or k.startswith("net.4.") for k in bc_sd.keys())
        ckpt_says_bn = has_bn_keys or has_bn_layers

    hidden_for_build = inferred_hidden if inferred_hidden is not None else args.hidden_size
    def build_model(use_bn: bool, obs_dim_: int, act_dim_guess: int):
        return (MLP if use_bn else MLP_no_batch_bias)(obs_dim_, act_dim_guess, chunk_len=args.chunk_len, hidden_size=hidden_for_build).to(device)

    if args.resume and os.path.exists(bc_path) and ckpt_says_bn is not None:
        use_bn_now = ckpt_says_bn
    else:
        use_bn_now = bool(args.use_bn_model)

    model = build_model(use_bn_now, obs_dim, act_dim or 6)

    if args.resume and os.path.exists(bc_path):
        sd = load_state(bc_path, map_location=device)
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            use_bn_now = not use_bn_now
            model = build_model(use_bn_now, obs_dim, act_dim or 6)
            model.load_state_dict(sd, strict=True)
        print(f"[BC] Loaded {bc_path} (obs_dim={obs_dim}, act_dim={act_dim}, hidden={hidden_for_build}, bn={use_bn_now})")
    else:
        loader = build_loader()
        sample_obs, sample_act = next(iter(loader))
        if act_dim is None:
            act_dim = int(sample_act.shape[1] if sample_act.ndim == 2 else sample_act.shape[2])
            model = build_model(use_bn_now, obs_dim, act_dim)
        model, bc_losses = train_bc(model, loader, device, args.lr, args.epochs)
        torch.save(model.state_dict(), bc_path); print(f"[BC] Saved {bc_path}")
        with open(os.path.join(args.artifacts_dir, 'bc_train_losses.json'), 'w') as f: json.dump(bc_losses, f)

    # Tap module
    tap_module = model.net[args.tap_index]

    # ---------- Load/collect activations & cached obs ----------
    have_acts = os.path.exists(acts_path)
    have_obs  = os.path.exists(cached_obs_path)

    if args.resume and have_acts and have_obs:
        acts = torch.load(acts_path, map_location='cpu')
        cached_obs = torch.load(cached_obs_path, map_location='cpu')
        print(f"[SAE] Loaded acts {acts.shape} and cached_obs {cached_obs.shape}")
    elif args.resume and args.policy is None:
        acts = None; cached_obs = None
        print("[SAE] Resume w/o --policy: skipping recollection of activations/obs.")
    else:
        loader = build_loader()
        acts = collect_activations(model, loader, device, tap_module, collect_max_batches=args.collect_max_batches)
        torch.save(acts, acts_path); print(f"[SAE] Saved activations -> {acts_path}")
        # cache obs aligned
        cache = []
        tap = ActivationTap(tap_module)
        model.eval()
        with torch.no_grad():
            for obs, _ in tqdm(loader, desc="Caching obs (aligned)"):
                cache.append(obs.cpu()); _ = model(obs.to(device))
        tap.remove()
        cached_obs = torch.cat(cache, 0)
        torch.save(cached_obs, cached_obs_path); print(f"[SAE] Saved cached_obs -> {cached_obs_path}")

    # ---------- Load/train SAE ----------
    sae_path_exists = os.path.exists(sae_path)
    if args.resume and sae_path_exists:
        ck = torch.load(sae_path, map_location=device)
        sae = TopKSAE(ck['d_in'], ck['d_latent'], k=ck['k']).to(device)
        sae.load_state_dict(ck['sae_state'])
        print(f"[SAE] Loaded {sae_path} (d_in={ck['d_in']}, d_latent={ck['d_latent']}, k={ck['k']})")
        d_in = ck['d_in']
    else:
        if acts is None:
            raise RuntimeError("Need activations to train SAE. Provide --policy once to collect, or run without --resume.")
        d_in = int(acts.shape[1])
        d_latent = args.sae_latent_mult * args.hidden_size
        sae = train_sae(acts, d_latent=d_latent, k=args.sae_topk, steps=args.sae_steps,
                        batch_size=args.sae_batch, lr=args.sae_lr, device=device)
        torch.save({"sae_state": sae.state_dict(), "d_in": d_in, "d_latent": d_latent,
                    "k": args.sae_topk, "tap_index": args.tap_index}, sae_path)
        print(f"[SAE] Saved {sae_path} (d_in={d_in}, d_latent={d_latent}, k={args.sae_topk})")

    # ---------- If --video_only and prior selection exists, reuse ----------
    if args.video_only and os.path.exists(sel_path):
        with open(sel_path, 'r') as f: sel = json.load(f)
        j_star = int(sel['j_star']); target_name = sel['target_name']; target_idx = int(sel['target_act_idx'])
        print(f"[Select] (video_only) j={j_star}, target={target_name}, idx={target_idx}")
        return render_video(args, model, sae, tap_module, j_star, target_idx, target_name, device)

    # ---------- Feature selection ----------
    if (os.path.exists(acts_path)) and (acts is None):
        acts = torch.load(acts_path, map_location='cpu')

    if acts is not None and os.path.exists(cached_obs_path):
        Z_t = encode_Z(sae, acts, device=device).numpy()  # [N, m]
        cached_obs = torch.load(cached_obs_path, map_location='cpu')
        # Labels = BC actions on these cached observations
        model.eval()
        with torch.no_grad():
            acts_pred = []
            bs = 1024
            for i in range(0, cached_obs.shape[0], bs):
                ob = cached_obs[i:i+bs].to(device)
                a = model(ob); acts_pred.append(a.cpu())
            acts_pred = torch.cat(acts_pred, 0).numpy()   # [N, act_dim]

        TARGET_ACT_IDX = int(args.target_act_idx)
        joint_names = [
            "right hip (thigh)", "right knee (leg)", "right ankle (foot)",
            "left hip (thigh_left)", "left knee (leg_left)", "left ankle (foot_left)"
        ]
        TARGET_NAME = joint_names[TARGET_ACT_IDX] if TARGET_ACT_IDX < len(joint_names) else f"act[{TARGET_ACT_IDX}]"

        # Ridge probe Z -> y
        y = acts_pred[:, TARGET_ACT_IDX]                 # [N]
        N, m = Z_t.shape
        lam = 1e-2
        Xb = np.concatenate([Z_t, np.ones((N,1))], axis=1)
        A = Xb.T @ Xb + lam * np.eye(m+1)
        w_full = np.linalg.solve(A, Xb.T @ y); w = w_full[:-1]
        j_star = int(np.argmax(np.abs(w))); w_mag = float(np.abs(w[j_star]))
        # Fallback: torso pitch if action probe weak
        if w_mag < 1e-3:
            print("[Select] Action probe weak; trying torso pitch fallback (obs[0])…")
            y2 = cached_obs[:, 0].numpy()
            w2_full = np.linalg.solve(A, Xb.T @ y2); w2 = w2_full[:-1]
            j_star = int(np.argmax(np.abs(w2))); TARGET_NAME = "torso pitch (obs[0])"
        print(f"[Select] j={j_star} for {TARGET_NAME}")
    else:
        print("[Select] No cached_obs/acts; choosing feature by decoder L2 norm.")
        Wdec = sae.decoder.weight.detach().cpu().numpy()   # [D, m]
        norms = np.linalg.norm(Wdec, axis=0)
        j_star = int(np.argmax(norms))
        TARGET_ACT_IDX = int(args.target_act_idx)
        TARGET_NAME = f"decoder-norm feature (j={j_star})"

    # Persist selection for --video_only
    with open(sel_path, 'w') as f:
        json.dump({"j_star": int(j_star),
                   "target_name": TARGET_NAME,
                   "target_act_idx": int(TARGET_ACT_IDX)}, f)

    # ---------- Render video (calibrated & gated) ----------
    render_video(args, model, sae, tap_module, j_star, TARGET_ACT_IDX, TARGET_NAME, device)


def render_video(args, model, sae, tap_module, j_star, target_act_idx, target_name, device):
    import gymnasium as gym
    env_b = gym.make(args.env, render_mode='rgb_array')
    env_p = gym.make(args.env, render_mode='rgb_array')
    env_n = gym.make(args.env, render_mode='rgb_array')
    ob_b, _ = env_b.reset(seed=123); ob_p, _ = env_p.reset(seed=123); ob_n, _ = env_n.reset(seed=123)

    Wdec = sae.decoder.weight.detach().to(device)  # [D, m]
    wj = Wdec[:, j_star]                           # [D]

    # ---- Calibrate alpha so panels don't saturate at ±1
    alpha_goal = 0.35  # target unclipped delta on the chosen action dim
    alpha_cal = calibrate_alpha(model, tap_module, sae, ob_b, wj, device, target_act_idx, target_delta=alpha_goal)
    alpha_used = float(args.alpha) if (alpha_cal is None or np.isnan(alpha_cal)) else float(alpha_cal)
    alpha_pos = +abs(alpha_used)
    alpha_neg = -abs(alpha_used)
    print(f"[Calibrate] Using α≈{alpha_used:.3f} (goal Δ≈{alpha_goal}) — fallback was {args.alpha:.3f}")

    print(f"[Video] Writing {args.outfile} … target={target_name}  j={j_star}")

    def step_with_edit(env, ob_np, alpha_value):
        ob_t = torch.from_numpy(ob_np).float().unsqueeze(0)
        _, edited = act_with_feature_edit(model, tap_module, sae, ob_t, wj, alpha_value, device)
        a = edited.squeeze(0).cpu().numpy()
        ob_next, rew, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
        frame = env.render()
        return ob_next, frame, a

    # Use context manager for writer; ensure envs close cleanly
    try:
        with imageio.get_writer(args.outfile, fps=args.fps) as writer:
            for t in tqdm(range(args.steps)):
                # baseline (no edit)
                ob_b, frame_b, a_b = step_with_edit(env_b, ob_b, alpha_value=0.0)

                # phase-aware gate: only apply edit when feature is naturally active
                act_score = feature_activity(model, tap_module, sae, ob_b, wj, device)
                gate = (act_score > 0.0)  # simple threshold; tweak as desired

                this_alpha_pos = alpha_pos if gate else 0.0
                this_alpha_neg = alpha_neg if gate else 0.0

                ob_p, frame_p, a_p = step_with_edit(env_p, ob_p, alpha_value=this_alpha_pos)
                ob_n, frame_n, a_n = step_with_edit(env_n, ob_n, alpha_value=this_alpha_neg)

                if t % 10 == 0:
                    def clipv(v): return np.clip(v, -1.0, 1.0)
                    an_c, ab_c, ap_c = clipv(a_n[target_act_idx]), clipv(a_b[target_act_idx]), clipv(a_p[target_act_idx])
                    print(f"[t={t:03d}] τ_{target_name} raw:  -α={a_n[target_act_idx]:+5.2f}  base={a_b[target_act_idx]:+5.2f}  +α={a_p[target_act_idx]:+5.2f} "
                          f"| clipped: -α={an_c:+4.2f} base={ab_c:+4.2f} +α={ap_c:+4.2f} | gate={gate}")

                H = min(frame_b.shape[0], frame_p.shape[0], frame_n.shape[0])
                W = min(frame_b.shape[1], frame_p.shape[1], frame_n.shape[1])
                panel = np.concatenate([frame_n[:H,:W], frame_b[:H,:W], frame_p[:H,:W]], axis=1)
                writer.append_data(panel.astype(np.uint8))

        print("[Video] Done. Panels: (-α) | (baseline) | (+α)")
    finally:
        # Explicitly close envs to avoid noisy EGL destructor errors
        try: env_b.close()
        except: pass
        try: env_p.close()
        except: pass
        try: env_n.close()
        except: pass


if __name__ == "__main__":
    main()
