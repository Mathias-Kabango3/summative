from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import DroneDeliveryEnv, EnvConfig


def make_env(seed: int = 42):
    def _fn():
        env = DroneDeliveryEnv(EnvConfig())
        env.reset(seed=seed)
        return Monitor(env)

    return _fn


def train(timesteps: int, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(seed)])
    eval_env = DummyVecEnv([make_env(seed + 1)])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
    )

    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=50_000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model.save(str(output_dir / "pg_final"))

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    metrics = {
        "algorithm": "ppo",
        "timesteps": timesteps,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "model_path": str(output_dir / "pg_final.zip"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on custom grid delivery environment")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="models/pg")
    args = parser.parse_args()

    train(args.timesteps, Path(args.output), args.seed)
