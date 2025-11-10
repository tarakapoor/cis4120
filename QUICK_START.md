# Quick Start Guide

## Step-by-Step: Using the Neural Network Debugger

### 1. Start the Backend Server

```bash
cd python
./start_server.sh
```

Server will start at `http://localhost:8000`

### 2. Start the Frontend

In a new terminal:

```bash
npm start
```

Frontend will start at `http://localhost:1234`

### 3. Upload Your Model

1. Click **"Select Model File"**
2. Choose `imitator.pt`
3. The network will be visualized with **5 layers**: [17, 32, 32, 32, 6]

### 4. Basic Weight Perturbation (Optional)

Once the model is loaded:
- **Choose tensor key**: Select which layer to perturb (e.g., `net.0.weight`)
- **Scale**: Multiply all weights by a factor (e.g., 0.9)
- **Add noise**: Add Gaussian noise (e.g., std=0.01)
- **Set value**: Set all weights to a constant (e.g., 0.0)

### 5. Run the Model

Click **"Run Model"** to:
- Save the current model state (with any perturbations)
- Run it in the `hard_stable` environment
- Collect 5 trajectories
- Display performance metrics (avg/max/min rewards, trajectory length)

**Note**: The first run might take a moment as it initializes the environment.

### 6. SAE Feature Analysis (Advanced)

#### A. Load the SAE

1. Scroll down to the **"SAE Feature Analysis"** section
2. Click **"Load SAE"**
3. When prompted, enter `.` (dot) to use files in the python/ directory
4. You should see: `SAE loaded! d_latent=128, k=16`

#### B. Understand Top Features

Once loaded, the UI automatically:
- Runs **ridge regression** to find which SAE features predict specific behaviors
- Displays the **top 10 most interpretable features**
- Shows their regression weights

**Example:**
```
Feature 42 (weight: 1.234)   ← Strongly predicts the target behavior
Feature 17 (weight: -0.987)  ← Negatively correlated with behavior
```

#### C. Perturb a Feature

1. **Select a feature** from the dropdown (default: top feature)
2. **Adjust alpha slider**:
   - `α > 0`: Amplify the feature (make behavior stronger)
   - `α < 0`: Suppress the feature (reduce behavior)
   - `α = 0`: No perturbation
3. **Click "Apply Feature Perturbation"**

You'll see a confirmation showing:
- Which feature was perturbed
- The alpha value used
- The decoder norm (magnitude of the feature direction)

#### D. Run and Compare

After applying feature perturbation:
1. Click **"Run Model"** again
2. Compare the new results with your previous baseline run
3. Observe how the specific feature affects the behavior

## What Each Feature Does

### Weight Perturbation
- **Direct manipulation** of network weights
- Affects all computations through that layer
- Less interpretable but powerful

### SAE Feature Perturbation
- **Semantic manipulation** of learned features
- Each feature corresponds to a specific behavior/concept
- More interpretable - you know what you're changing

## Example Workflow

```
1. Upload imitator.pt
2. Click "Run Model" → baseline: avg_reward = 150
3. Load SAE (enter ".")
4. Feature 42 is selected (top feature)
5. Set alpha = 3.0 (amplify)
6. Click "Apply Feature Perturbation"
7. Click "Run Model" → perturbed: avg_reward = 180 (improved!)
8. Try alpha = -3.0 (suppress)
9. Click "Run Model" → suppressed: avg_reward = 120 (worse)
```

This shows that Feature 42 positively contributes to performance.

## Troubleshooting

### "Run Model" shows no results

Check the browser console (F12) for errors. The backend logs will show what went wrong.

### SAE fails to load

Make sure these files exist in `python/`:
- `walker_sae.pt`
- `cached_obs.pt`
- `tapped_activations.pt`

### Feature perturbation seems to have no effect

- Try larger alpha values (±5 instead of ±2)
- Make sure the feature is actually active for your input distribution
- Check that `tap_index` matches the layer used during SAE training (default: 4)

## Tips

1. **Start with baseline**: Always run the model without perturbations first
2. **Small alpha first**: Start with α=±1, then increase if needed
3. **Compare multiple features**: Try the top 3-5 features to see which has the strongest effect
4. **Visualize the network**: Click neurons in the graph to see connections
5. **Check layer structure**: Make sure all 5 layers are visible (fixed bug!)

## What's Next?

After you're comfortable with basic perturbations:
- Try different target dimensions in feature interpretation
- Experiment with combining weight + feature perturbations
- Use different tap layers (change `tap_index`)
- Visualize feature activations across the dataset
