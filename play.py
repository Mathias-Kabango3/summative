"""
play.py — Load the best trained DQN model and play in the Atari environment.

Loads dqn_model.zip and runs the agent using a greedy Q-policy
(deterministic=True) to maximize performance. Displays the game via env.render().
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import importlib
import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor


# ─── Environment Utilities ───

def _try_register_ale_envs():
    try:
        ale_py = importlib.import_module("ale_py")
        if hasattr(gym, "register_envs"):
            gym.register_envs(ale_py)
        elif hasattr(ale_py, "register_envs"):
            ale_py.register_envs(gym)
    except Exception:
        pass


def _is_env_registered(env_id):
    try:
        gym.spec(env_id)
        return True
    except Exception:
        return False


def resolve_env_id(env_id):
    candidates = [env_id]
    if env_id.startswith("ALE/") and env_id.endswith("-v5"):
        game = env_id.split("/", 1)[1].replace("-v5", "")
        candidates.append(f"{game}NoFrameskip-v4")
    if env_id.startswith("ALE/"):
        _try_register_ale_envs()
    for candidate in candidates:
        if _is_env_registered(candidate):
            return candidate
    raise RuntimeError(f"Environment '{env_id}' was not found.")


def build_play_env(env_id, policy, frame_stack):
    """Create an environment for playing with human rendering."""
    if policy == "CnnPolicy":
        env = make_atari_env(
            env_id, n_envs=1, seed=123,
            env_kwargs={"obs_type": "rgb", "render_mode": "human"},
        )
        env = VecMonitor(env)
        env = VecFrameStack(env, n_stack=frame_stack)
        return env
    env = make_vec_env(
        env_id, n_envs=1, seed=123,
        env_kwargs={"obs_type": "ram", "render_mode": "human"},
    )
    env = VecMonitor(env)
    return env


# ─── Main ───

if __name__ == "__main__":
    # Member configurations
    members = {
        "mathias": {
            "name": "Mathias Kabango",
            "model": Path("runs_member1_dqn/dqn_model.zip"),
            "metadata": Path("runs_member1_dqn/best_model_metadata.json"),
        },
        "kellen": {
            "name": "Kellen Murerwa",
            "model": Path("runs_member2_dqn/dqn_model.zip"),
            "metadata": Path("runs_member2_dqn/best_model_metadata.json"),
        },
        "edine": {
            "name": "Edine Noella Mugunga",
            "model": Path("runs_member3_dqn/dqn_model.zip"),
            "metadata": Path("runs_member3_dqn/best_model_metadata.json"),
        },
        "best": {
            "name": "Best Overall",
            "model": Path("dqn_model.zip"),
            "metadata": Path("best_overall_model/best_model_metadata.json"),
        },
    }

    # Pick member from command line: python play.py mathias
    member_key = sys.argv[1] if len(sys.argv) > 1 else "best"
    if member_key not in members:
        print(f"Usage: python play.py [{' | '.join(members.keys())}]")
        sys.exit(1)

    member = members[member_key]
    model_path = member["model"]
    metadata_path = member["metadata"]
    episodes = 20
    frame_stack = 4

    print(f"\n{'='*50}")
    print(f"  Playing: {member['name']}")
    print(f"{'='*50}")

    # Load metadata to determine policy and env
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        policy = meta.get("policy", "CnnPolicy")
        env_id = resolve_env_id(meta.get("env_id", "ALE/Breakout-v5"))
        print(f"Policy: {policy}, Env: {env_id}")
        print(f"Hyperparameters: {meta.get('hyperparameters', {})}")
        print(f"Training mean reward: {meta.get('mean_reward', 'N/A')}")
    else:
        policy = "CnnPolicy"
        env_id = resolve_env_id("ALE/Breakout-v5")
        print(f"No metadata found. Using defaults: policy={policy}, env={env_id}")

    # Build environment and load model
    env = build_play_env(env_id, policy, frame_stack)
    model = DQN.load(str(model_path), env=env)

    # Run episodes with greedy Q-policy (deterministic=True)
    total_rewards = []
    for episode in range(episodes):
        obs = env.reset()
        done = [False]
        episode_reward = 0.0

        while not done[0]:
            # Greedy Q-policy: select action with highest Q-value
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = env.step(action)
            episode_reward += float(rewards[0])
            env.render()

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: reward={episode_reward:.2f}")

    env.close()
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")
