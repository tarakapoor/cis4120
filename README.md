# Neural Network Debugger & Visualizer

An interactive web application for visualizing, debugging, and manipulating neural network models with an intuitive interface. Upload PyTorch models, explore network structures, adjust weights, and analyze model behavior through sparse autoencoder (SAE) features.

## Features

- 🎨 **Interactive Network Visualization**: Visualize neural network architectures with clickable neurons and weighted connections
- ⚙️ **Weight Adjustment**: Adjust individual weights between neurons to steer model behavior
- 📊 **Model Execution**: Run models in various environments and view performance metrics
- 🔍 **SAE Feature Analysis**: Load sparse autoencoders to interpret and perturb learned features
- 📚 **Inline Definitions**: Click on technical terms to see beginner-friendly explanations
- 🎯 **Real-time Feedback**: See weight changes and activations update in real-time

## Prerequisites

- **Node.js** (v14 or higher) and **npm**
- **Python** (3.8 or higher)
- **Conda** (for managing Python environment)
- **PyTorch** model files (`.pt` format)
- **SAE files** (optional, for feature analysis)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/tarakapoor/cis4120.git
cd cis4120
```

### 2. Backend Setup (Python/FastAPI)

#### Create and Activate Conda Environment

```bash
# Create conda environment (if not already created)
conda create -n ddpm_bc_env python=3.8
conda activate ddpm_bc_env

# Navigate to python directory
cd python

# Install dependencies
pip install -r requirements.txt
```

The backend requires FastAPI, uvicorn, and python-multipart, which are included in `requirements.txt`.

#### Start the Backend Server

```bash
# Make sure you're in the python directory
cd python

# Option 1: Use the start script (recommended)
# Make the script executable if needed:
chmod +x start_server.sh
./start_server.sh

# Option 2: Start manually
conda activate ddpm_bc_env
uvicorn server:app --reload --port 8000
```

The backend server will start at `http://localhost:8000`.

### 3. Frontend Setup (React/Node)

#### Install Dependencies

```bash
# From the project root directory
npm install
```

This will install all required dependencies including:
- React
- TypeScript
- D3.js (for visualizations)
- Parcel (for bundling)

#### Start the Frontend

```bash
npm start
```

The frontend will start at `http://localhost:1234` (or similar port) and automatically open in your browser.

## Usage Guide

### Step 1: Upload a Model

1. In the web interface, click **"Select Model File"** in the left panel
2. Choose your PyTorch model file (e.g., `imitator.pt`)
   - You can upload from any location on your computer
   - Example model file: `python/imitator.pt` (if located in the python directory)
3. The network structure will be automatically inferred and visualized

**Supported formats:**
- `.pt` files (PyTorch checkpoint files)
- `.json` files (manual network structure definitions)

### Step 2: Explore the Network Visualization

Once your model is loaded:

- **Click on neurons** (circles) to select them and see their connections
- **Selected neurons** are highlighted in red
- **Connected neurons** are highlighted in orange
- **Edges (connections)** show weights:
  - Blue edges = positive weights (strengthen connections)
  - Red edges = negative weights (weaken/invert connections)
  - Thicker edges = stronger weights
- **Zoom and pan** by scrolling and dragging on the visualization
- **Hover over neurons** to see their IDs

### Step 3: Adjust Weights

1. **Click on a neuron** to select it
2. The **Weight Adjustment Panel** will appear on the right
3. Use the sliders to adjust individual weights to connected neurons
4. Watch the visualization update in real-time:
   - Edge colors change based on weight values
   - Edge thickness reflects weight magnitude
   - Before/After values show your changes

**Weight Operations:**
- **Scale**: Multiply all weights in a layer by a factor
- **Add Noise**: Add Gaussian noise to weights
- **Set Value**: Set all weights to a constant value

### Step 4: Run the Model

1. **Select an environment** from the dropdown (e.g., `Walker2d-v4`, `hard_stable`)
2. **Enable "Capture activations"** if you want to see activation visualizations
3. Click **"Run Model"** to execute the model with current weights
4. View results:
   - Average reward
   - Max/Min rewards
   - Average trajectory length
   - Number of trajectories

### Step 5: Load Sparse Autoencoder (SAE) for Feature Analysis

1. **Ensure you have SAE files** in the `python/` directory:
   - `walker_sae.pt` (the SAE model)
   - `cached_obs.pt` (cached observations, optional)
   - `tapped_activations.pt` (cached activations, optional)

2. **Click "Load SAE"** in the "SAE Feature Analysis" section
3. When prompted, enter `.` (dot) to use files in the `python/` directory
4. The SAE will load and automatically analyze top interpretable features

### Step 6: Perturb SAE Features

1. **Select a feature** from the "Top Interpretable Features" dropdown
2. **Adjust the alpha slider**:
   - `α > 0`: Amplify the feature (make behavior stronger)
   - `α < 0`: Suppress the feature (reduce behavior)
   - `α = 0`: No perturbation
3. **Click "Apply Feature Perturbation"**
4. **Run the model again** to see how the feature perturbation affects behavior

## Understanding Technical Terms

The interface includes **inline definitions** for technical jargon:

- **Click on any underlined term** (like "neural network", "neuron", "weight", "policy") to see a beginner-friendly explanation
- Terms are defined with examples to help you understand concepts
- Related terms are linked for easy exploration

**Common terms:**
- **Neuron**: A basic processing unit in the network (shown as a circle)
- **Weight**: The strength of a connection between neurons
- **Layer**: A group of neurons that process information at the same stage
- **Activation**: The output value of a neuron after processing input
- **Policy**: The strategy that determines what action to take
- **Perturbation**: A small change to weights to see how it affects behavior
- **Steering**: Adjusting weights to guide network behavior

## Example Workflow

```
1. Upload imitator.pt
   → Network visualized with 5 layers: [17, 32, 32, 32, 6]

2. Click "Run Model" (baseline)
   → Average reward: 150

3. Load SAE (enter ".")
   → SAE loaded! d_latent=128, k=16
   → Top feature: Feature 42 (weight: 1.234)

4. Select Feature 42, set alpha = 3.0 (amplify)
   → Click "Apply Feature Perturbation"

5. Click "Run Model" (perturbed)
   → Average reward: 180 (improved!)

6. Try alpha = -3.0 (suppress)
   → Click "Run Model"
   → Average reward: 120 (worse)

This shows that Feature 42 positively contributes to performance.
```

## File Structure

```
cis4120/
├── src/                          # Frontend source code
│   ├── App.tsx                   # Main application component
│   ├── components/
│   │   ├── NeuralNetworkVisualizer/
│   │   │   ├── ActivationViewer.tsx
│   │   │   ├── ModelUpload.tsx
│   │   │   ├── NetworkGraph.tsx
│   │   │   └── WeightAdjustmentPanel.tsx
│   │   └── UI/
│   │       ├── InfoPanel.tsx
│   │       └── TermDefinition.tsx
│   ├── data/
│   │   └── glossary.ts           # Technical term definitions
│   └── utils/
│       └── modelUtils.ts
├── python/                       # Backend server
│   ├── server.py                 # FastAPI server
│   ├── start_server.sh           # Server startup script
│   ├── requirements.txt          # Python dependencies
│   ├── imitator.pt               # Example model file
│   ├── walker_sae.pt             # Example SAE file
│   ├── cached_obs.pt             # Cached observations
│   └── tapped_activations.pt     # Cached activations
├── package.json                  # Node.js dependencies
├── tsconfig.json                 # TypeScript configuration
└── README.md                     # This file
```

## Troubleshooting

### Backend Issues

**Problem: Backend won't start**
- Make sure conda environment is activated: `conda activate ddpm_bc_env`
- Check if port 8000 is available: `lsof -i :8000`
- Verify FastAPI dependencies are installed: `pip install fastapi uvicorn python-multipart`

**Problem: SAE fails to load**
- Ensure these files exist in `python/`:
  - `walker_sae.pt`
  - `cached_obs.pt` (optional but recommended)
  - `tapped_activations.pt` (optional but recommended)
- Check file paths when prompted (enter `.` for current directory)

### Frontend Issues

**Problem: Frontend can't connect to backend**
- Verify backend is running at `http://localhost:8000`
- Check browser console (F12) for CORS errors
- Ensure both frontend and backend are running simultaneously

**Problem: Model upload fails**
- Verify the `.pt` file contains a valid PyTorch state_dict
- Check browser console for error messages
- Ensure the model architecture matches expected format

### Model Execution Issues

**Problem: "Run Model" shows no results**
- Check browser console (F12) for errors
- Verify backend logs for detailed error messages
- Ensure the model is compatible with the selected environment
- Check that the conda environment has all required dependencies

**Problem: Feature perturbation seems to have no effect**
- Try larger alpha values (±5 instead of ±2)
- Ensure the feature is actually active for your input distribution
- Verify that `tap_index` matches the layer used during SAE training (default: 4)
- Check that cached data exists and is loaded correctly

### Visualization Issues

**Problem: Network graph doesn't display**
- Refresh the page after uploading the model
- Check browser console for JavaScript errors
- Verify that the model structure was correctly inferred

**Problem: Weight adjustments don't update visualization**
- Ensure a neuron is selected before adjusting weights
- Check that the Weight Adjustment Panel is visible
- Try clicking a different neuron and adjusting again

## Tips for Best Results

1. **Start with baseline**: Always run the model without perturbations first to establish a baseline
2. **Small changes first**: Start with small weight adjustments (α=±1) before trying larger values
3. **Compare multiple features**: Try the top 3-5 features to see which has the strongest effect
4. **Use inline definitions**: Click on technical terms to understand concepts as you explore
5. **Check layer structure**: Verify all layers are visible in the visualization
6. **Experiment iteratively**: Make small changes, run the model, observe results, then adjust

## API Endpoints

The backend provides these endpoints:

- `POST /upload` - Upload .pt or .json file
- `GET /model/{id}/summary` - Get model architecture
- `POST /model/{id}/perturb` - Perturb weights (scale, add_noise, set)
- `POST /model/{id}/edit_at` - Edit individual weight values
- `POST /model/{id}/save_and_run` - Save and run model
- `POST /model/{id}/load_sae` - Load sparse autoencoder
- `POST /model/{id}/interpret_features` - Analyze SAE features
- `POST /model/{id}/sae_perturb` - Apply feature perturbation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Support

For issues, questions, or contributions, please open an issue on the repository.

