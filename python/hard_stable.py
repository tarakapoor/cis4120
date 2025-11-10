import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import gymnasium as gym
from gymnasium import spaces
from typing import Sequence, Any, Callable, Tuple, Optional, Dict


def make_challenging_pair(mu: float = 1/4) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Creates two pairs of (A, K) matrices for the environment dynamics.
    """
    c_mu = 3/2 * mu
    A_1 = np.array([
        [1 + mu, c_mu],
        [-c_mu, 1 - 2 * mu]
    ], dtype=np.float32)
    A_2 = np.array([
        [-(1 - mu / 4), c_mu],
        [0, 1 - 2 * mu]
    ], dtype=np.float32)
    K_1 = np.array([
        [-(1 + mu), -c_mu],
        [c_mu, 0]
    ], dtype=np.float32)
    K_2 = np.array([
        [(1 - mu / 4), -c_mu],
        [0, 0]
    ], dtype=np.float32)
    return (A_1, K_1), (A_2, K_2)

def default_bump(x: np.ndarray) -> float:
    """
    A bump function used for restricting perturbations.
    """
    x_norm = np.linalg.norm(x)
    if x_norm < 1:
        return 1.0
    elif x_norm > 2:
        return 0.0
    else:
        return float(np.exp(-1 / (1 - (x_norm - 1)**2)))


class PerturbationModel(nn.Module):
    """
    A PyTorch module to model perturbations.
    """
    def __init__(self, features: Sequence[int], d_in: int):
        super().__init__()
        layers = []
        current_dim = d_in
        for f_out in features:
            linear_layer = nn.Linear(current_dim, f_out)
            init.xavier_normal_(linear_layer.weight)
            init.uniform_(linear_layer.bias, a=-1.0, b=1.0)
            layers.append(linear_layer)
            layers.append(nn.Tanh())
            current_dim = f_out
        
        self.hidden_layers = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(current_dim, 1)
        init.xavier_normal_(self.output_layer.weight)
        init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x.squeeze() / 200.0

class EmbedGymEnv(gym.Env):
    """
    The challenging environment implemented in the Farama Gymnasium interface.
    """
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self, A: np.ndarray, K: np.ndarray, d: int,
                 model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
                 tau: float = 0.001, delta: float = 0.3, omega: float = 1.0,
                 bump_fn: Callable[[np.ndarray], float] = default_bump,
                 perturb_model_features: Sequence[int] = (16, 16)):
        super().__init__()

        self.A = A
        self.K = K
        self.d = d
        self.tau = tau
        self.omega = omega
        self.bump_fn = bump_fn

        # Setup the perturbation model
        self.perturbation_model = PerturbationModel(features=perturb_model_features, d_in=d)
        if model_state_dict:
            self.perturbation_model.load_state_dict(model_state_dict)
        self.perturbation_model.eval() # Set to evaluation mode

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + d,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + d,), dtype=np.float32)

        self.state: Optional[np.ndarray] = None
        self.np_random: Optional[np.random.Generator] = None # Gymnasium RNG

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None, "State is not initialized. Call reset() first."
        return self.state.astype(np.float32)

    def _get_info(self) -> Dict:
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        Z = self.np_random.binomial(1, 0.5)

        # Generate z (uniform over scaled sphere)
        z_normal = self.np_random.normal(size=(self.d,))
        z_norm = np.linalg.norm(z_normal)
        if z_norm == 0:
            z_unit = np.zeros_like(z_normal)
        else:
            z_unit = z_normal / z_norm
        z_scaled = z_unit * self.np_random.uniform()
        
        z_full = np.concatenate((np.zeros(2, dtype=np.float32), z_scaled.astype(np.float32)))
        z_full[2] += 3.0

        # Generate w (uniform over scaled sphere, d+1 components for w)
        w_normal = self.np_random.normal(size=(self.d + 1,))
        w_norm = np.linalg.norm(w_normal)
        if w_norm == 0:
            w_unit = np.zeros_like(w_normal)
        else:
            w_unit = w_normal / w_norm
        w_scaled = w_unit * self.np_random.uniform()

        w_full = np.concatenate((np.zeros(1, dtype=np.float32), w_scaled.astype(np.float32)))
        

        if Z == 0:
            self.state = z_full
        else: # Z == 1
            self.state = w_full


        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")
        
        action_np = np.array(action, dtype=np.float32)

        # Prepare input for perturbation model
        state_for_model_np = self.state[2:]
        state_for_model_torch = torch.tensor(state_for_model_np, dtype=torch.float32)
        
        with torch.no_grad():
            g_out = self.perturbation_model(state_for_model_torch).item()

        # Calculate restrict term
        state_offset_for_bump = self.state.copy()
        state_offset_for_bump[2] -= 3.0
        restrict = self.bump_fn(state_offset_for_bump)

        # Calculate perturbation
        perturbation_val = (
            -self.tau * restrict * g_out
            + self.omega * (self.tau**2) * restrict * g_out
            - self.tau * action_np[0] * self.bump_fn(action_np)
        )

        # Update lower part of the state
        state_lower_prev = self.state[:2]
        intermediate_state_lower = self.A @ state_lower_prev
        intermediate_state_lower[0] += perturbation_val
        
        state_intermediate_for_next = np.concatenate(
            (intermediate_state_lower, np.zeros(self.d, dtype=np.float32))
        )

        # Apply action to the state components
        next_state_candidate = state_intermediate_for_next.copy()
        next_state_candidate[:2] += action_np[:2]

        # Clip to prevent NaN explosion
        self.state = np.clip(next_state_candidate, -1_000_000_000, 1_000_000_000)

        # Calculate reward: -abs(next_state[0])
        reward = -float(np.abs(self.state[0]))
        
        terminated = False
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        raise NotImplementedError("Rendering is not supported.")

    def close(self):
        pass

class EmbedExpertTorch:
    def __init__(self, K: np.ndarray, model_state_dict: Dict[str, torch.Tensor],
                 tau: float, d_model_input: int,
                 bump_fn: Callable[[np.ndarray], float] = default_bump,
                 perturb_model_features: Sequence[int] = (16, 16)):
        self.K = K
        self.tau = tau
        self.bump_fn = bump_fn
        
        self.perturbation_model = PerturbationModel(features=perturb_model_features, d_in=d_model_input)
        self.perturbation_model.load_state_dict(model_state_dict)
        self.perturbation_model.eval()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs_np = np.array(observation, dtype=np.float32)
        
        action_dynamic_part = self.K @ obs_np[:2]
        
        action_candidate = np.concatenate(
            (action_dynamic_part, np.zeros(obs_np.shape[0] - 2, dtype=np.float32))
        )

        # Perturbation model part
        obs_for_model_np = obs_np[2:]
        obs_for_model_torch = torch.tensor(obs_for_model_np, dtype=torch.float32)
        
        with torch.no_grad():
            g_out = self.perturbation_model(obs_for_model_torch).item()

        obs_offset_for_bump = obs_np.copy()
        obs_offset_for_bump[2] -= 3.0
        restrict = self.bump_fn(obs_offset_for_bump)
        
        expert_perturbation = self.tau * restrict * g_out
        
        action_candidate[0] += expert_perturbation
        
        return action_candidate

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:

        action = self(observation)
        return action, None

def create_gym_environment(d: int, pair_first: bool,
                           pytorch_seed: Optional[int] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None
                           ) -> EmbedGymEnv:
    """
    Creates an instance of the EmbedGymEnv.
    Optionally seeds PyTorch for reproducible model initialization.
    """
    if pytorch_seed is not None:
        torch.manual_seed(pytorch_seed)

    (A1, K1), (A2, K2) = make_challenging_pair()
    A, K_matrix = (A1, K1) if pair_first else (A2, K2)

    if state_dict is None:
        temp_model = PerturbationModel(features=(16, 16), d_in=d)
        model_state_dict = temp_model.state_dict()
    else:
        model_state_dict = state_dict


    env = EmbedGymEnv(A=A, K=K_matrix, d=d, model_state_dict=model_state_dict)
    return env



if __name__ == '__main__':
    print("Creating and testing the Gym environment...")
    pytorch_seed = 117
    if pytorch_seed is not None:
        torch.manual_seed(pytorch_seed) # Ensure same init if creating new
    dimension = 4
    # Create environment (using the first matrix pair)
    # Seed PyTorch for reproducible model weights if desired
    env_instance = create_gym_environment(d=dimension, pair_first=True, pytorch_seed=pytorch_seed)
    
    # Test reset
    print("Resetting environment...")
    obs, info = env_instance.reset(seed=42) # Seed the environment's RNG
    print(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Initial observation: {obs[:5]}")

    # Test step
    print("\nStepping through environment with random actions...")
    for i in range(5):
        action = env_instance.action_space.sample() # Sample a random action
        obs, reward, terminated, truncated, info = env_instance.step(action)
        print(f"Step {i+1}:")
        print(f"  Observation (first 5 elements): {obs[:5]}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            print("Episode finished.")
            break
    
    env_instance.close()
    print("\nEnvironment test finished.")


    print("\nCreating and testing the Expert policy...")
    # For the expert, we need a model_state_dict.
    # We can get it from the environment's model if it's the same,
    # or create a new one if the expert uses a separate model instance.
    # Assuming the expert uses the same model parameters as initialized for the env:
    
    # Re-create a model instance to get state_dict, or get from env if accessible
    expert_model_for_expert = PerturbationModel(features=(16,16), d_in=dimension)
    expert_model_state_dict = expert_model_for_expert.state_dict()
    
    # The K matrix for the expert comes from make_challenging_pair
    (_, K_expert_matrix), _ = make_challenging_pair() # Example K1
    if not env_instance.K.flags['OWNDATA']: # K from env might be a view
        K_expert_matrix = env_instance.K.copy()
    else:
        K_expert_matrix = env_instance.K


    expert_policy = EmbedExpertTorch(
        K=K_expert_matrix, 
        model_state_dict=expert_model_state_dict,
        tau=env_instance.tau,
        d_model_input=dimension
    )

    # Roll out 5 sample trajectories
    print("\nRolling out 5 sample trajectories of length 5...")
    for traj in range(5):
        print(f"\nTrajectory {traj + 1}:")
        obs, _ = env_instance.reset()
        action = expert_policy(obs)
        print(f"Initial state: {obs}, \nInitial action: {action}")
        
        # for step in range(5):
        #     action = expert_policy(obs)
        #     obs, reward, terminated, truncated, _ = env_instance.step(action)
        #     if terminated or truncated:
        #         break

    # Roll out expert policy for 30 steps
    current_obs, _ = env_instance.reset()
    first_action = expert_policy(current_obs)
    print(f"\nFirst expert action: {first_action}")
    
    # Run for 30 steps
    for step in range(30):
        action = expert_policy(current_obs)
        current_obs, reward, terminated, truncated, info = env_instance.step(action)
        if step == 29:  # Last step
            print(f"\nFinal expert action: {action}")
            # Calculate squared norm of final state
            state_norm = np.linalg.norm(current_obs)**2
            print(f"\nSquared norm of state at step 30: {state_norm:.6f}")
        if terminated or truncated:
            break
            
    print("\nExpert policy rollout finished.")