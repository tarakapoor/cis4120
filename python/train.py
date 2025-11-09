# bc_with_sae_walker.py
# Behavior Cloning + Sparse Autoencoder (SAE) on hidden activations (Walker or any MuJoCo env)
# - Trains BC MLP
# - Collects activations from a chosen internal layer (pre-GELU)
# - Trains a Top-K Sparse Autoencoder
# - Saves: BC weights, SAE weights, activations snapshot, simple probes

import os
import argparse
import json
import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---- Your environment helpers (must exist in your codebase) ----
from rollout import TrajLoader, CollectTrajs

# ----------------------------
# 1) Models: BC MLP variants
# ----------------------------
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

    def forward(self, x):
        return self.net(x)

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

    def forward(self, x):
        return self.net(x)

# ----------------------------
# 2) BC training
# ----------------------------
def train_bc(model, loader, device, lr, epochs):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    model.train()
    train_losses = []

    for ep in tqdm(range(epochs), desc="BC training", leave=True, position=0):
        total_loss = 0.0
        for obs, acts in loader:
            obs = obs.to(device)
            acts = acts.to(device)
            preds = model(obs)
            loss = criterion(preds, acts.flatten(start_dim=1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * obs.size(0)
        scheduler.step()
        avg_loss = total_loss / len(loader.dataset)
        train_losses.append(avg_loss)

    print(f"[BC] Final loss: {train_losses[-1]:.6f}")
    return model, train_losses

# -----------------------------------------
# 3) Hook util to tap a chosen internal layer
# -----------------------------------------
class ActivationTap:
    def __init__(self, module):
        self.buffer = []
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out is [B, D]
        self.buffer.append(out.detach().float().cpu())

    def stacked(self):
        if len(self.buffer) == 0:
            return torch.empty(0)
        return torch.cat(self.buffer, dim=0)

    def remove(self):
        self.h.remove()

@torch.no_grad()
def collect_layer_activations(model: nn.Module,
                              loader: DataLoader,
                              device: str,
                              tap_module: nn.Module,
                              max_batches: Optional[int] = None) -> torch.Tensor:
    model.eval()
    tap = ActivationTap(tap_module)
    n_batches = 0
    for obs, acts in tqdm(loader, desc="Collecting activations"):
        obs = obs.to(device)
        _ = model(obs)  # forward triggers hook
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    acts = tap.stacked()  # [N, D]
    tap.remove()
    return acts

# ----------------------------
# 4) Sparse Autoencoder (Top-K)
# ----------------------------
class TopKSAE(nn.Module):
    """
    Simple, stable SAE:
    - Encoder: Linear, ReLU
    - Sparsity: per-row Top-K
    - Decoder: Linear
    """
    def __init__(self, d_in: int, d_latent: int, k: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.k = k
        self.encoder = nn.Linear(d_in, d_latent, bias=False)
        self.decoder = nn.Linear(d_latent, d_in, bias=False)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def encode(self, x):
        z = F.relu(self.encoder(x))
        if self.k is not None and self.k < z.shape[1]:
            vals, idx = torch.topk(z, k=self.k, dim=1)
            mask = torch.zeros_like(z)
            mask.scatter_(1, idx, 1.0)
            z = z * mask
        return z

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

def sae_recon_loss(x, x_hat):
    return F.mse_loss(x_hat, x)

def train_sae(acts: torch.Tensor,
              d_latent: int,
              k: int,
              steps: int,
              batch_size: int,
              lr: float,
              device: str) -> TopKSAE:
    dataset = TensorDataset(acts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    d_in = acts.shape[1]
    sae = TopKSAE(d_in, d_latent, k=k).to(device)
    opt = optim.AdamW(sae.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=steps)

    sae.train()
    it = iter(loader)
    pbar = tqdm(range(steps), desc="SAE training")

    for step in pbar:
        try:
            (xb,) = next(it)
        except StopIteration:
            it = iter(loader)
            (xb,) = next(it)
        xb = xb.to(device)
        xh, _ = sae(xb)
        loss = sae_recon_loss(xb, xh)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            pbar.set_postfix(loss=float(loss.item()))
    sae.eval()
    return sae

# ----------------------------
# 5) Probing helpers
# ----------------------------
@torch.no_grad()
def get_latents(sae: TopKSAE, X: torch.Tensor, device: str) -> torch.Tensor:
    sae.eval()
    Z = sae.encode(X.to(device))
    return Z.detach().cpu()

def quick_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    x, y: 1D tensors (same length, on CPU)
    """
    xv = x - x.mean()
    yv = y - y.mean()
    denom = (xv.norm() * yv.norm()).item()
    if denom == 0:
        return 0.0
    return float((xv @ yv).item() / denom)

# ----------------------------
# 6) Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="BC + SAE on hidden activations")
    # BC / data
    parser.add_argument('--env', type=str, required=True, help='Gym env ID, e.g., Walker2d-v4')
    parser.add_argument('--policy', type=str, required=True, help='Path to expert policy checkpoint for rollout')
    parser.add_argument('--chunk_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser.add_argument('--prop_noised', type=float, default=0.5)
    parser.add_argument('--max_timesteps', type=int, default=300)
    parser.add_argument('--num_traj', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--use_bn_model', action='store_true', help='Use MLP with BatchNorm (default). If false, use biasless MLP.')

    # SAE
    parser.add_argument('--tap_index', type=int, default=4,
                        help='Index in model.net to tap (e.g., 4 is BN after 2nd Linear for BN model).')
    parser.add_argument('--sae_latent_mult', type=int, default=4, help='latent size = mult * hidden_size')
    parser.add_argument('--sae_topk', type=int, default=16)
    parser.add_argument('--sae_steps', type=int, default=50000)
    parser.add_argument('--sae_batch', type=int, default=1024)
    parser.add_argument('--sae_lr', type=float, default=1e-3)
    parser.add_argument('--collect_max_batches', type=int, default=None,
                        help='Limit activation collection to this many batches (for quick tests).')

    # Probing demo
    parser.add_argument('--run_probe_demo', action='store_true',
                        help='Runs a small correlation probe against a simple target (norm of obs).')

    args = parser.parse_args()
    device = args.device

    # ---- Collect trajectories from expert to form BC dataset ----
    trajs = CollectTrajs(
        env_name=args.env,
        policy_path=args.policy,
        max_timesteps=args.max_timesteps,
        num_trajectories=args.num_traj,
        noise_scale=args.noise_scale,
        prop_noised=args.prop_noised,
        deterministic=True,
        seed=None
    )

    loader = TrajLoader(
        trajs,
        chunk_len=args.chunk_len,
        batch_size=args.batch_size,
        device=args.device
    )

    # infer dims
    sample_obs, sample_act = next(iter(loader))
    obs_dim = sample_obs.shape[1]
    act_dim = sample_act.shape[1] if sample_act.ndim == 2 else sample_act.shape[2]

    # ---- Build & train BC model ----
    if args.use_bn_model:
        model = MLP(obs_dim, act_dim, chunk_len=args.chunk_len, hidden_size=args.hidden_size)
    else:
        model = MLP_no_batch_bias(obs_dim, act_dim, chunk_len=args.chunk_len, hidden_size=args.hidden_size)

    model, train_losses = train_bc(model, loader, device, args.lr, args.epochs)
    torch.save(model.state_dict(), 'imitator.pt')
    with open('bc_train_losses.json', 'w') as f:
        json.dump(train_losses, f)
    print("[BC] Saved to imitator.pt")

    # ---- Choose layer to tap ----
    # For BN model, self.net = [0:Linear, 1:BN, 2:GELU, 3:Linear, 4:BN, 5:GELU, 6:Linear, 7:GELU, 8:Linear]
    # tap_index=4 is the BN after the 2nd Linear (a good pre-GELU representation).
    try:
        tap_module = model.net[args.tap_index]
    except Exception as e:
        raise ValueError(f"tap_index {args.tap_index} invalid for this architecture") from e

    # ---- Collect activations at tapped layer ----
    acts = collect_layer_activations(
        model=model,
        loader=loader,
        device=device,
        tap_module=tap_module,
        max_batches=args.collect_max_batches
    )
    if acts.ndim != 2:
        raise RuntimeError(f"Expected collected activations to be [N, D], got {acts.shape}")
    torch.save(acts, 'tapped_activations.pt')
    print(f"[SAE] Collected activations: {acts.shape}, saved to tapped_activations.pt")

    # ---- Train SAE on activations ----
    d_in = acts.shape[1]
    d_latent = args.sae_latent_mult * args.hidden_size  # usually 4x hidden
    sae = train_sae(
        acts=acts,
        d_latent=d_latent,
        k=args.sae_topk,
        steps=args.sae_steps,
        batch_size=args.sae_batch,
        lr=args.sae_lr,
        device=device
    )
    torch.save({
        "sae_state": sae.state_dict(),
        "d_in": d_in,
        "d_latent": d_latent,
        "k": args.sae_topk,
        "tap_index": args.tap_index
    }, 'walker_sae.pt')
    print(f"[SAE] Saved to walker_sae.pt (d_in={d_in}, d_latent={d_latent}, k={args.sae_topk})")

    # ---- Optional quick probe demo ----
    if args.run_probe_demo:
        # Example target: ||obs||_2 per sample (simple, always available)
        # NOTE: This is just to illustrate the probing API; replace with real labels like contact flags, phase of gait, etc.
        print("[Probe] Running quick demo probe vs ||obs||")
        # Recollect obs to align with acts rows (we do a pass identical to collect_layer_activations)
        # To stay simple, we recompute and push obs into a list in the same traversal order.
        model.eval()
        obs_stream = []
        tap = ActivationTap(tap_module)
        with torch.no_grad():
            for obs, acts_true in tqdm(loader, desc="Collecting obs for probe"):
                obs_stream.append(obs.cpu())
                _ = model(obs.to(device))
        tap.remove()
        obs_mat = torch.cat(obs_stream, dim=0)  # [N, obs_dim]
        y = obs_mat.norm(dim=1)                 # [N]

        Z = get_latents(sae, acts, device=device)  # [N, m]
        # single-feature correlations
        m = Z.shape[1]
        cors = []
        y_cpu = y.cpu()
        for j in range(m):
            r = quick_corrcoef(Z[:, j], y_cpu)
            cors.append((r, j))
        cors.sort(reverse=True, key=lambda x: abs(x[0]))

        topk_show = 10
        print(f"[Probe] Top {topk_show} |corr(feature_j, ||obs||)|:")
        for r, j in cors[:topk_show]:
            print(f"  j={j:5d}  corr={r:+.4f}")

        # Save top feature list
        with open('probe_top_features.json', 'w') as f:
            json.dump([{"feature": int(j), "corr": float(r)} for r, j in cors[:100]], f)
        print("[Probe] Saved probe_top_features.json")

    print("Done.")

if __name__ == '__main__':
    main()
