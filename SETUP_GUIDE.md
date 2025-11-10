# Neural Network Visualizer - Setup Guide

## Overview
This webapp allows you to:
1. Upload PyTorch model files (.pt)
2. Visualize the network structure
3. Interactively perturb weights
4. Run the model and see results

## Setup Instructions

### 1. Backend Setup (Python/FastAPI)

First, make sure you have the conda environment with all dependencies:

```bash
# Activate your conda environment
conda activate ddpm_bc_env

# Install the additional FastAPI dependencies
cd python
pip install fastapi==0.115.0 uvicorn==0.34.0 python-multipart==0.0.20
```

### 2. Start the Backend Server

```bash
# From the python directory
cd python
./start_server.sh
```

Or manually:
```bash
conda activate ddpm_bc_env
cd python
uvicorn server:app --reload --port 8000
```

The server will start at `http://localhost:8000`

### 3. Frontend Setup (React/Node)

In a new terminal:

```bash
# From the project root
npm install
npm start
```

The frontend will start at `http://localhost:1234` (or similar)

## Usage

### 1. Upload a Model

- Click "Select Model File" and choose a `.pt` file
- The backend will automatically:
  - Load the model's state_dict
  - Infer the network architecture
  - Display the network visualization

### 2. Perturb Weights

Once a model is loaded, you can:
- **Scale**: Multiply all weights by a factor (e.g., 0.9 to reduce magnitude)
- **Add Noise**: Add Gaussian noise to weights
- **Set Value**: Set all weights to a constant

Choose a tensor key (like "net.0.weight") and click the desired operation.

### 3. Run the Model

After loading (and optionally perturbing) the model:
1. Click "Run Model"
2. The backend will:
   - Save the current model state
   - Run it in the `hard_stable` environment
   - Collect 5 trajectories
   - Return performance metrics

Results will show:
- Average reward
- Max/Min rewards
- Average trajectory length

## API Endpoints

The backend provides these endpoints:

- `POST /upload` - Upload .pt or .json file
- `GET /model/{id}/summary` - Get model architecture
- `POST /model/{id}/perturb` - Perturb weights (scale, add_noise, set)
- `POST /model/{id}/edit_at` - Edit individual weight values
- `POST /model/{id}/save_and_run` - Save and run model

## Troubleshooting

### Backend won't start
- Make sure conda environment is activated: `conda activate ddpm_bc_env`
- Check if port 8000 is available: `lsof -i :8000`
- Install FastAPI dependencies: `pip install fastapi uvicorn python-multipart`

### Frontend can't connect to backend
- Verify backend is running at `http://localhost:8000`
- Check browser console for CORS errors
- Make sure both frontend and backend are running

### Model run fails
- Check that the .pt file contains a compatible model format
- Ensure the model architecture matches what the environment expects
- Look at the server console output for detailed error messages

## Next Steps

After confirming the basic functionality works:
1. You can add more SAE-based functionality
2. Customize the environment parameters
3. Add more visualization features
4. Implement weight editing at individual neuron level
