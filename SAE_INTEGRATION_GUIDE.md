# SAE Integration Guide

## Overview

The webapp now supports **Sparse Autoencoder (SAE)** based interpretation and perturbation, similar to the functionality in `sae.py`. This allows you to:

1. **Interpret** which SAE features correspond to specific behaviors
2. **Perturb** learned features (instead of raw weights)
3. **Run** the model with SAE-based interventions

## Prerequisites

You need SAE artifacts created by running the training script. These should be in a directory (e.g., `artifacts_w32/`) containing:

- `walker_sae.pt` - Trained SAE checkpoint
- `cached_obs.pt` - Observations used for training
- `tapped_activations.pt` - Activations from the tapped layer

### Creating SAE Artifacts

Run the SAE training script with your model:

```bash
python bc_sae_walker_demo.py \
  --env hard_stable \
  --policy imitator.pt \
  --artifacts_dir artifacts_w32 \
  --hidden_size 32 \
  --tap_index 4
```

## Using SAE in the Web UI

### 1. Upload Your Model

- Upload your `.pt` file (e.g., `imitator.pt`)
- The model will be loaded and visualized

### 2. Load SAE

- Click **"Load SAE"** in the SAE Feature Analysis section
- Enter your artifacts directory path (e.g., `artifacts_w32`)
- The SAE will be loaded with:
  - Decoder/encoder weights
  - Cached observations and activations for interpretation

### 3. Interpret Features

Once the SAE is loaded, the UI automatically:
- Runs ridge regression to find which SAE features predict specific action dimensions
- Displays top 10 most interpretable features
- Shows their regression weights

### 4. Perturb Features

- Select a feature from the dropdown (sorted by interpretability)
- Adjust the **alpha slider** to control perturbation strength
  - Positive α: Amplify the feature
  - Negative α: Suppress the feature
- Click **"Apply Feature Perturbation"**

### 5. Run and Compare

- After applying perturbations, click **"Run Model"**
- Compare results with/without perturbations

## Backend API Endpoints

### POST `/model/{id}/load_sae`

Load SAE artifacts from a directory.

**Request:**
```json
{
  "artifacts_dir": "artifacts_w32",
  "tap_index": 4
}
```

**Response:**
```json
{
  "ok": true,
  "d_in": 32,
  "d_latent": 128,
  "k": 16,
  "tap_index": 4,
  "has_cached_data": true
}
```

### POST `/model/{id}/interpret_features`

Find most interpretable SAE features for a target action dimension.

**Request:**
```json
{
  "target_dim": 0,
  "top_k": 10
}
```

**Response:**
```json
{
  "features": [
    {"feature_idx": 42, "weight": 1.234, "abs_weight": 1.234},
    {"feature_idx": 17, "weight": -0.987, "abs_weight": 0.987},
    ...
  ],
  "target_dim": 0
}
```

### POST `/model/{id}/sae_perturb`

Apply SAE feature-based perturbation.

**Request:**
```json
{
  "feature_idx": 42,
  "alpha": 2.0
}
```

**Response:**
```json
{
  "ok": true,
  "feature_idx": 42,
  "alpha": 2.0,
  "decoder_norm": 0.534,
  "message": "Feature perturbation parameters stored"
}
```

## How It Works

### SAE Feature Perturbation

Unlike raw weight perturbation, SAE perturbation works at the **activation level**:

1. **Forward pass**: Run model to get activations at tapped layer
2. **Get decoder direction**: Extract decoder weight vector `w_j` for feature `j`
3. **Perturb activations**: `x_edited = x_baseline + α * w_j`
4. **Continue forward**: Use perturbed activations for rest of forward pass

This is more interpretable than weight perturbation because SAE features often correspond to meaningful behaviors (e.g., "torso lean", "leg swing").

### Feature Interpretation

Uses ridge regression to find features that predict specific outputs:

```
Z @ w = y
```

Where:
- `Z`: SAE feature activations [N x d_latent]
- `y`: Target behavior (e.g., action[0]) [N]
- `w`: Feature importances [d_latent]

Features with high `|w_j|` strongly predict the target behavior.

## Troubleshooting

### "SAE not found" error

Make sure your artifacts directory contains `walker_sae.pt`. Check the path.

### "Need SAE + cached data" error

The artifacts directory needs all three files:
- `walker_sae.pt`
- `cached_obs.pt`
- `tapped_activations.pt`

Run the full SAE training pipeline to generate these.

### Feature perturbation has no effect

- Make sure `tap_index` matches the layer used during SAE training
- Try larger alpha values (±3 to ±5)
- Check that the feature is actually active for your input distribution

## Next Steps

- **Visualize feature activations**: Show which inputs activate each feature
- **Side-by-side videos**: Like in `sae.py`, render videos with +α, baseline, -α
- **Automatic alpha calibration**: Auto-tune α to achieve desired effect size
- **Feature clustering**: Group related features by decoder similarity
