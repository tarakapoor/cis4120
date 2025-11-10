# Activation Viewer Guide

## Overview

The Activation Viewer lets you "open up the policy and watch it work" by visualizing how signals propagate through the network in real-time during a rollout.

## Features

### ✅ Time-Based Activation Visualization
- **Captures activations** for each timestep during model execution
- **Play/pause controls** to watch the network in action
- **Scrubbing timeline** to jump to any timestep
- **Variable playback speed** (0.5x, 1x, 2x, 5x)

### ✅ Layer-by-Layer Display
- Shows activations for **all linear layers**
- Color-coded neurons:
  - **Blue intensity**: Positive activations
  - **Red intensity**: Negative activations
  - **Brightness**: Activation magnitude
- Hover over neurons to see exact values

### ✅ Input/Output Display
- **Observation**: Current state (17-dim for Walker2d)
- **Action**: Model output (6-dim for Walker2d)
- Updates in real-time as you scrub through timesteps

## How to Use

### 1. Enable Activation Capture

When running your model:
1. Check the **"Capture activations"** checkbox
2. Click **"Run Model"**

**Note**: This only captures the first trajectory to keep data manageable.

### 2. View the Activation Viewer

Once the run completes, the Activation Viewer appears below the network graph showing:
- Timeline scrubber
- Play/pause controls
- Current observation and action
- Layer activations

### 3. Navigate Through Time

**Controls:**
- **⏮ Reset**: Jump to timestep 0
- **◀ Step**: Go back one timestep
- **▶ Play / ⏸ Pause**: Auto-advance through timesteps
- **Step ▶**: Go forward one timestep
- **Speed dropdown**: Change playback rate

**Timeline Scrubber:**
- Drag the slider to any timestep
- Progress bar shows current position

### 4. Understand What You See

#### Observation Panel (Blue)
Shows the current state of the environment:
- Position, velocity, joint angles
- Updates each timestep as the agent moves

#### Action Panel (Green)
Shows what the model decided to do:
- Torques for each joint
- Direct output from the final layer

#### Layer Activation Panels (Yellow)
Shows internal computations:
- **layer_0**: First hidden layer (17 → 32)
- **layer_2**: Second hidden layer (32 → 32)
- **layer_4**: Third hidden layer (32 → 32)
- **layer_6**: Output layer (32 → 6)

Each small box is a neuron, numbered 0-31 (or 0-5 for output).

**Color meaning:**
- Bright blue: Strong positive activation
- Bright red: Strong negative activation
- Faint/white: Near-zero activation

### 5. Debug Your Policy

Use the viewer to:

**Find problems:**
- **Dead neurons**: Always near-zero (white) → not learning
- **Saturated neurons**: Always max brightness → may need regularization
- **Erratic patterns**: Jumping randomly → instability

**Understand behavior:**
- **Which neurons activate** when the agent does specific actions
- **How information flows** from observation through layers to action
- **Temporal patterns**: Do certain neurons activate rhythmically?

**Compare runs:**
- Run baseline model → note activation patterns
- Perturb weights/features → run again
- Compare: what changed?

## Example Workflow

```
1. Upload imitator.pt
2. Enable "Capture activations"
3. Click "Run Model"
4. Wait for run to complete (~10-30 seconds)
5. Activation Viewer appears
6. Click "Play" to watch the agent
7. Notice: when agent leans left, layer_2 neurons 5-8 light up
8. Pause at interesting moments
9. Scrub back and forth to understand
10. Take notes on patterns
```

## Technical Details

### Data Structure

Each timestep contains:
```json
{
  "timestep": 42,
  "observation": [0.1, 0.2, ...],  // 17 values
  "action": [0.5, -0.3, ...],       // 6 values
  "layer_0": [[...]],               // 32 neurons
  "layer_2": [[...]],               // 32 neurons
  "layer_4": [[...]],               // 32 neurons
  "layer_6": [[...]]                // 6 neurons (=action)
}
```

### Performance

- Only the **first trajectory** is captured (saves memory)
- Typical size: ~300 timesteps × 5 layers × 32 neurons = ~50KB
- Scrubbing is instant (all in browser memory)
- No network requests during playback

### Limitations

- Currently only captures first trajectory
- Max 300 timesteps per run
- Doesn't show gradients (forward pass only)
- No comparison view (yet)

## Tips

1. **Start slow**: Use 0.5x speed to catch details
2. **Find patterns**: Look for neurons that activate together
3. **Compare layers**: How does info transform layer-to-layer?
4. **Mark interesting timesteps**: Note when behavior changes
5. **Use with perturbations**: Run baseline → perturb → compare activations

## Troubleshooting

### "No activation data available"

Make sure:
- "Capture activations" checkbox was **checked**
- Model run **completed successfully**
- Environment is **Walker2d-v4** (or compatible)

### Viewer is laggy

- Reduce playback speed
- Close other browser tabs
- Refresh page and reload model

### Wrong number of layers showing

Check that your model architecture matches:
- 4 linear layers (net.0, net.2, net.4, net.6)
- If different, activation viewer adapts automatically

## Future Enhancements

Potential additions:
- Compare multiple runs side-by-side
- Highlight neurons that changed after perturbation
- Export activation data as CSV
- Show gradient flow (backward pass)
- Integrate with SAE feature visualization
- 3D trajectory visualization
