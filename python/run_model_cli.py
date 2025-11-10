#!/usr/bin/env python3
"""
CLI wrapper for running models with rollout.py
Usage: python run_model_cli.py --policy <path_to_pt> --env <env_name> [--num_traj N] [--max_steps N]
"""
import argparse
from rollout import CollectTrajs
import json

def main():
    parser = argparse.ArgumentParser(description='Run a policy and collect trajectories')
    parser.add_argument('--policy', type=str, required=True, help='Path to the .pt policy file')
    parser.add_argument('--env', type=str, default='hard_stable', help='Environment name (default: hard_stable)')
    parser.add_argument('--num_traj', type=int, default=5, help='Number of trajectories to collect')
    parser.add_argument('--max_steps', type=int, default=300, help='Max steps per trajectory')
    parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic actions (1=True, 0=False)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    print(f"Running policy from {args.policy} in environment {args.env}")
    print(f"Collecting {args.num_traj} trajectories with max {args.max_steps} steps each")

    trajectories = CollectTrajs(
        env_name=args.env,
        policy_path=args.policy,
        max_timesteps=args.max_steps,
        num_trajectories=args.num_traj,
        deterministic=bool(args.deterministic),
        seed=args.seed
    )

    # Calculate statistics
    total_rewards = [sum(traj['rewards']) for traj in trajectories]
    avg_reward = sum(total_rewards) / len(total_rewards)
    max_reward = max(total_rewards)
    min_reward = min(total_rewards)

    traj_lengths = [len(traj['rewards']) for traj in trajectories]
    avg_length = sum(traj_lengths) / len(traj_lengths)

    results = {
        'num_trajectories': len(trajectories),
        'avg_reward': float(avg_reward),
        'max_reward': float(max_reward),
        'min_reward': float(min_reward),
        'avg_length': float(avg_length),
        'trajectory_rewards': [float(r) for r in total_rewards],
        'trajectory_lengths': traj_lengths
    }

    print("\nResults:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Max reward: {max_reward:.2f}")
    print(f"  Min reward: {min_reward:.2f}")
    print(f"  Average trajectory length: {avg_length:.2f}")
    print(f"\nJSON Output:")
    print(json.dumps(results, indent=2))

    return results

if __name__ == '__main__':
    main()
