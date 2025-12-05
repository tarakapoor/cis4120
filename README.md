# Neural Network Debugger & Visualizer

An interactive web application for visualizing, debugging, editing, and interpreting neural network models. The tool provides an intuitive UI for exploring PyTorch model architectures, adjusting weights, analyzing activations, and integrating Sparse Autoencoders (SAEs) for mechanistic interpretability.

The goal of this project is to make neural networks transparent, understandable, and steerable in real time — with a frontend built in React + TypeScript + D3.js and a backend implemented in FastAPI + PyTorch.

---

# Core Features

## Interactive Neural Network Visualization
- Automatically infer MLP architectures from uploaded PyTorch `.pt` or `.json` model files  
- Display neurons, layers, and weighted edges using D3 force-directed layout  
- Zoom, pan, and click to select neurons  
- Color-coded edges:
  - **Blue** → positive weights  
  - **Grey** → negative weights  
- Edge thickness reflects magnitude  
- Real-time updates as weights are modified

## Weight Editing & Steering Interface
- Select neurons and inspect incoming/outgoing weights  
- Adjust weights with sliders  
- Bulk operations:
  - Scale weights  
  - Add Gaussian noise  
  - Set weight values to constants  
- Visualization updates instantly

## Activation Viewer
- Capture activations during model execution  
- Visualize neuron-by-neuron activations across layers  
- Compare baseline vs perturbed activations

## Model Execution Environment
- Run RL policy models (e.g., Walker2d-v4) directly from UI  
- Automatic rollout execution in backend  
- View metrics:
  - average reward  
  - max/min reward  
  - trajectory length  
  - number of samples  

## SAE (Sparse Autoencoder) Integration
- Upload SAE models (`walker_sae.pt`)  
- Load cached activations and observations  
- Compute top interpretable SAE features  
- Apply **feature-level perturbations**:
  - α > 0 → amplify feature  
  - α < 0 → suppress feature  
- Rerun model to evaluate causal effects of features

## Inline Glossary
- Hover or click on underlined technical terms  
- Beginner-friendly definitions stored in `src/data/glossary.ts`  
- Helps non-experts understand neural network concepts

---

# Prerequisites

- **Node.js** ≥ 14  
- **npm**  
- **Python** ≥ 3.8  
- **Conda** (recommended)  
- **PyTorch** installed in your environment  

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/tarakapoor/cis4120.git
cd cis4120
```
## 2. Backend Setup

```bash
conda create -n ddpm_bc_env python=3.8
conda activate ddpm_bc_env

cd python
pip install -r requirements.txt
```
Start the Backend Server
Option A — using script (recommended)
```bash
cd python
chmod +x start_server.sh
./start_server.sh
```
Option B — manually
```bash
uvicorn server:app --reload --port 8000
```

## 3. Frontend Setup

Backend runs at http://localhost:8000

```bash
npm install
npm start
```
The frontend will open at: http://localhost:1234
(Or next available port)



# Usage Guide
## Step 1 — Upload a Model
- Click Select Model File
- Upload .pt (PyTorch state_dict) or .json architecture
- Model is visualized automatically

## Step 2 — Explore Visualization
- Click neurons to inspect
- Orange → connected neurons
- Scroll → zoom
- Drag → pan

## Step 3 — Edit Weights
- Adjust 1-to-1 weight sliders
- Apply bulk ops:
   - scale
   - noise
   - set constant
Graph updates instantly.

## Step 4 — Run Model
- Choose environment (Walker2d-v4, hard_stable, etc.)
- Optionally enable activation capture
- Click Run Model
- View rollout metrics

## Step 5 — Load SAE
- Ensure files exist in /python:
- walker_sae.pt
- cached_obs.pt
- tapped_activations.pt

Click Load SAE, enter . for current directory.

## Step 6 — Feature Perturbation
Select SAE feature
Set α
Apply perturbation
Run model to compare output




# Project Structure
```text
cis4120/
├── python/
│   ├── server.py
│   ├── sae.py
│   ├── rollout.py
│   ├── run_model_cli.py
│   ├── run_bc_model.py
│   ├── hard_stable.py
│   ├── cached_obs.pt
│   ├── tapped_activations.pt
│   ├── walker_sae.pt
│   ├── imitator.pt
│   ├── requirements.txt
│   └── start_server.sh
│
├── src/
│   ├── App.tsx
│   ├── index.tsx
│   ├── index.html
│   ├── components/
│   │   ├── NeuralNetworkVisualizer/
│   │   │   ├── ActivationViewer.tsx
│   │   │   ├── ModelUpload.tsx
│   │   │   ├── NetworkGraph.tsx
│   │   │   └── WeightAdjustmentPanel.tsx
│   │   └── UI/
│   │       ├── DynamicBackground.tsx
│   │       ├── HelpPage.tsx
│   │       ├── InfoPanel.tsx
│   │       └── TermDefinition.tsx
│   ├── data/
│   │   └── glossary.ts
│   └── utils/
│       └── modelUtils.ts
│
├── package.json
├── tsconfig.json
├── QUICK_START.md
├── SETUP_GUIDE.md
├── FINAL_SUMMARY.md
└── README.md
```


# Troubleshooting
## Model Upload Fails
Ensure .pt is a valid PyTorch state_dict
Check backend logs
Confirm architecture parsing supports your model
## Frontend Cannot Reach Backend
Backend must run on port 8000
Check CORS errors in browser console
Ensure Conda env has required packages
## SAE Fails to Load
Verify files exist in /python
Use . when asked for path
## Graph Not Updating
Reselect neuron
Refresh UI
Check devtools console for TypeScript errors

# AI-Assisted Code Attribution
Project code was developed using the help of ChatGPT, Cursor, and Claude for code generation, scaffolding, debugging, and refactoring.
All AI-assisted code was reviewed, edited, and adapted by our project team.
No tutorials, templates, or external starter code were used.
