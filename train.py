"""
train.py — Train a DQN agent on ALE/Breakout-v5 using Stable Baselines3.

Trains 30 experiments across 3 group members:
  - Mathias Kabango: CnnPolicy (10 experiments)
  - Kellen Murerwa: CnnPolicy (10 experiments)
  - Edine Noella Mugunga: MlpPolicy (10 experiments)

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import importlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor


# ─── Data Structures ───

@dataclass
class HyperParams:
    lr: float
    gamma: float
    batch_size: int
    epsilon_start: float
    epsilon_end: float
    epsilon_fraction: float


class EpisodeLoggerCallback(BaseCallback):
    """Logs per-episode reward and length to a CSV file."""
    def __init__(self, csv_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rows = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.rows.append({
                    "timesteps": int(self.num_timesteps),
                    "reward": float(ep["r"]),
                    "episode_length": int(ep["l"]),
                })
        return True

    def _on_training_end(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timesteps", "reward", "episode_length"])
            writer.writeheader()
            writer.writerows(self.rows)


# ─── Environment Utilities ───

def _is_env_registered(env_id: str) -> bool:
    try:
        gym.spec(env_id)
        return True
    except Exception:
        return False


def _try_register_ale_envs() -> bool:
    try:
        ale_py = importlib.import_module("ale_py")
    except Exception:
        return False
    try:
        if hasattr(gym, "register_envs"):
            gym.register_envs(ale_py)
            return True
        if hasattr(ale_py, "register_envs"):
            ale_py.register_envs(gym)
            return True
    except Exception:
        return False
    return False


def resolve_env_id(env_id: str) -> str:
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


def build_env(env_id, policy, n_envs, seed, frame_stack, render_mode=None):
    """Create a vectorized Atari environment with appropriate wrappers."""
    if policy == "CnnPolicy":
        env = make_atari_env(
            env_id, n_envs=n_envs, seed=seed,
            env_kwargs={"obs_type": "rgb", "render_mode": render_mode},
        )
        env = VecMonitor(env)
        env = VecFrameStack(env, n_stack=frame_stack)
        return env
    env = make_vec_env(
        env_id, n_envs=n_envs, seed=seed,
        env_kwargs={"obs_type": "ram", "render_mode": render_mode},
        wrapper_class=Monitor,
    )
    env = VecMonitor(env)
    return env


# ─── Training Functions ───

def train_once(env_id, policy, hparams, total_timesteps, n_envs, frame_stack,
               seed, eval_freq, eval_episodes, device, run_dir):
    """Train a single DQN experiment and return metadata dict."""
    run_dir.mkdir(parents=True, exist_ok=True)
    train_env = build_env(env_id, policy, n_envs, seed, frame_stack)
    eval_env = build_env(env_id, policy, 1, seed + 100, frame_stack)

    episode_logger = EpisodeLoggerCallback(run_dir / "episode_log.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=eval_freq, deterministic=True, render=False,
        n_eval_episodes=eval_episodes,
    )

    model = DQN(
        policy, train_env,
        learning_rate=hparams.lr, gamma=hparams.gamma, batch_size=hparams.batch_size,
        exploration_initial_eps=hparams.epsilon_start,
        exploration_final_eps=hparams.epsilon_end,
        exploration_fraction=hparams.epsilon_fraction,
        buffer_size=100_000, learning_starts=10_000,
        target_update_interval=10_000, train_freq=4, gradient_steps=1,
        optimize_memory_usage=False, max_grad_norm=10,
        verbose=1, seed=seed,
        tensorboard_log=str(run_dir / "tb"), device=device,
    )

    model.learn(total_timesteps=total_timesteps,
                callback=[episode_logger, eval_callback],
                progress_bar=True)
    model.save(run_dir / "dqn_model")

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )

    metadata = {
        "env_id": env_id, "policy": policy,
        "obs_type": "rgb" if policy == "CnnPolicy" else "ram",
        "hyperparameters": asdict(hparams),
        "mean_reward": float(mean_reward), "std_reward": float(std_reward),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    train_env.close()
    eval_env.close()
    return metadata


def run_member_experiments(member_name, experiments, policy, output_dir,
                           env_id="ALE/Breakout-v5", total_timesteps=1_000_000,
                           n_envs=4, frame_stack=4, seed=42,
                           eval_freq=20_000, eval_episodes=5, device="auto"):
    """Run all experiments for one group member and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_env_id(env_id)
    if resolved != env_id:
        print(f"Using compatible env: {resolved}")

    summary_rows = []
    best = None

    for idx, hp in enumerate(experiments, start=1):
        run_name = f"{member_name}_exp{idx}"
        run_dir = output_dir / run_name
        print(f"\n{'='*60}")
        print(f"  {member_name} | Experiment {idx}/10 | {policy}")
        print(f"  lr={hp.lr}, gamma={hp.gamma}, batch={hp.batch_size}, "
              f"eps=[{hp.epsilon_start}->{hp.epsilon_end}], frac={hp.epsilon_fraction}")
        print(f"{'='*60}")

        metadata = train_once(
            env_id=resolved, policy=policy, hparams=hp,
            total_timesteps=total_timesteps, n_envs=n_envs, frame_stack=frame_stack,
            seed=seed + idx, eval_freq=eval_freq, eval_episodes=eval_episodes,
            device=device, run_dir=run_dir,
        )
        summary_rows.append({
            "experiment": idx, "member": member_name, "policy": policy,
            "lr": hp.lr, "gamma": hp.gamma, "batch_size": hp.batch_size,
            "epsilon_start": hp.epsilon_start, "epsilon_end": hp.epsilon_end,
            "epsilon_fraction": hp.epsilon_fraction,
            "mean_reward": metadata["mean_reward"], "std_reward": metadata["std_reward"],
        })
        if best is None or metadata["mean_reward"] > best["mean_reward"]:
            best = {"mean_reward": metadata["mean_reward"], "run_dir": run_dir, "metadata": metadata}

    # Save summary CSV
    fieldnames = ["experiment", "member", "policy", "lr", "gamma", "batch_size",
                  "epsilon_start", "epsilon_end", "epsilon_fraction", "mean_reward", "std_reward"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save best model
    if best is not None:
        best_model = best["run_dir"] / "dqn_model.zip"
        if best_model.exists():
            (output_dir / "dqn_model.zip").write_bytes(best_model.read_bytes())
        with (output_dir / "best_model_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(best["metadata"], f, indent=2)

    print(f"\n{member_name} done! Best mean reward: {best['mean_reward']:.2f}")
    return summary_rows, best


# ─── Main ───

if __name__ == "__main__":
    import shutil
    import matplotlib.pyplot as plt

    # Global config
    ENV_ID = "ALE/Breakout-v5"
    TOTAL_TIMESTEPS = 100_000
    N_ENVS = 8
    FRAME_STACK = 4
    SEED = 42
    EVAL_FREQ = 100_000
    EVAL_EPISODES = 3
    DEVICE = "auto"

    # ─── Mathias Kabango: CnnPolicy (10 experiments) ───
    member1_experiments = [
        HyperParams(2.5e-4, 0.99,  32, 1.0, 0.10, 0.10),   # 1: baseline
        HyperParams(1e-4,   0.99,  32, 1.0, 0.10, 0.10),   # 2: lower lr
        HyperParams(5e-4,   0.99,  32, 1.0, 0.10, 0.10),   # 3: higher lr
        HyperParams(1e-4,   0.95,  32, 1.0, 0.10, 0.10),   # 4: lower gamma
        HyperParams(1e-4,   0.999, 32, 1.0, 0.10, 0.10),   # 5: higher gamma
        HyperParams(1e-4,   0.99,  64, 1.0, 0.10, 0.10),   # 6: larger batch
        HyperParams(1e-4,   0.99,  32, 1.0, 0.05, 0.10),   # 7: lower eps_end
        HyperParams(1e-4,   0.99,  32, 1.0, 0.10, 0.20),   # 8: longer exploration
        HyperParams(1e-4,   0.99,  32, 1.0, 0.02, 0.25),   # 9: very low eps_end + long explore
        HyperParams(5e-5,   0.995, 64, 1.0, 0.05, 0.15),   # 10: conservative combo
    ]

    member1_rows, member1_best = run_member_experiments(
        member_name="Mathias_Kabango",
        experiments=member1_experiments,
        policy="CnnPolicy",
        output_dir="runs_mathias_dqn",
        env_id=ENV_ID, total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS, frame_stack=FRAME_STACK, seed=SEED,
        eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, device=DEVICE,
    )

    # ─── Kellen Murerwa: CnnPolicy (10 experiments) ───
    member2_experiments = [
        HyperParams(3e-4,   0.99,  32, 1.0, 0.10, 0.10),   # 1: slightly higher lr baseline
        HyperParams(3e-4,   0.99,  32, 1.0, 0.01, 0.10),   # 2: very low eps_end
        HyperParams(3e-4,   0.99,  32, 0.5, 0.10, 0.10),   # 3: lower eps_start
        HyperParams(3e-4,   0.99,  32, 1.0, 0.10, 0.30),   # 4: very long exploration phase
        HyperParams(3e-4,   0.99,  32, 1.0, 0.10, 0.05),   # 5: very short exploration phase
        HyperParams(3e-4,   0.99, 128, 1.0, 0.10, 0.10),   # 6: large batch
        HyperParams(3e-4,   0.99,  16, 1.0, 0.10, 0.10),   # 7: small batch
        HyperParams(3e-4,   0.98,  32, 1.0, 0.10, 0.10),   # 8: lower gamma
        HyperParams(3e-4,   0.995, 32, 1.0, 0.10, 0.10),   # 9: higher gamma
        HyperParams(2e-4,   0.99,  64, 1.0, 0.05, 0.15),   # 10: balanced combo
    ]

    member2_rows, member2_best = run_member_experiments(
        member_name="Kellen_Murerwa",
        experiments=member2_experiments,
        policy="CnnPolicy",
        output_dir="runs_kellen_dqn",
        env_id=ENV_ID, total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS, frame_stack=FRAME_STACK, seed=SEED + 100,
        eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, device=DEVICE,
    )

    # ─── Edine Noella Mugunga: MlpPolicy (10 experiments) ───
    member3_experiments = [
        HyperParams(1e-4,   0.99,  32, 1.0, 0.10, 0.10),   # 1: baseline MLP
        HyperParams(5e-4,   0.99,  32, 1.0, 0.10, 0.10),   # 2: higher lr
        HyperParams(1e-3,   0.99,  32, 1.0, 0.10, 0.10),   # 3: aggressive lr
        HyperParams(1e-4,   0.99,  64, 1.0, 0.10, 0.10),   # 4: larger batch
        HyperParams(1e-4,   0.99, 128, 1.0, 0.10, 0.10),   # 5: very large batch
        HyperParams(1e-4,   0.95,  32, 1.0, 0.10, 0.10),   # 6: low gamma
        HyperParams(1e-4,   0.999, 32, 1.0, 0.10, 0.10),   # 7: high gamma
        HyperParams(1e-4,   0.99,  32, 1.0, 0.01, 0.30),   # 8: low eps_end + long explore
        HyperParams(1e-4,   0.99,  32, 1.0, 0.20, 0.05),   # 9: high eps_end + short explore
        HyperParams(2.5e-4, 0.995, 64, 1.0, 0.05, 0.15),   # 10: best-guess combo
    ]

    member3_rows, member3_best = run_member_experiments(
        member_name="Edine_Noella_Mugunga",
        experiments=member3_experiments,
        policy="MlpPolicy",
        output_dir="runs_edine_dqn",
        env_id=ENV_ID, total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS, frame_stack=FRAME_STACK, seed=SEED + 200,
        eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, device=DEVICE,
    )

    # ─── Comparison & Best Model Selection ───

    all_rows = member1_rows + member2_rows + member3_rows
    df_all = pd.DataFrame(all_rows) if "pd" in dir() else None

    # Print results
    import pandas as pd
    df_all = pd.DataFrame(all_rows)

    for member in ["Mathias_Kabango", "Kellen_Murerwa", "Edine_Noella_Mugunga"]:
        df_m = df_all[df_all["member"] == member]
        print(f"\n{'='*70}")
        print(f"  {member} — Hyperparameter Tuning Results")
        print(f"{'='*70}")
        print(df_m[["experiment", "policy", "lr", "gamma", "batch_size",
                     "epsilon_start", "epsilon_end", "epsilon_fraction",
                     "mean_reward", "std_reward"]].to_string(index=False))

    # CnnPolicy vs MlpPolicy
    print(f"\n{'='*70}")
    print("  CnnPolicy vs MlpPolicy Comparison")
    print(f"{'='*70}")
    policy_stats = df_all.groupby("policy").agg(
        avg_mean_reward=("mean_reward", "mean"),
        best_mean_reward=("mean_reward", "max"),
        worst_mean_reward=("mean_reward", "min"),
        num_experiments=("mean_reward", "count"),
    ).reset_index()
    print(policy_stats.to_string(index=False))

    # Best per member
    print(f"\n{'='*70}")
    print("  Best Experiment Per Member")
    print(f"{'='*70}")
    best_per_member = df_all.loc[df_all.groupby("member")["mean_reward"].idxmax()]
    print(best_per_member[["member", "experiment", "policy", "lr", "gamma",
                           "batch_size", "epsilon_end", "epsilon_fraction",
                           "mean_reward", "std_reward"]].to_string(index=False))

    # Overall best model
    candidates = [
        ("Mathias_Kabango", member1_best, "runs_mathias_dqn"),
        ("Kellen_Murerwa", member2_best, "runs_kellen_dqn"),
        ("Edine_Noella_Mugunga", member3_best, "runs_edine_dqn"),
    ]
    overall_best = max(candidates, key=lambda c: c[1]["mean_reward"])
    winner_name, winner_data, winner_dir_name = overall_best

    print(f"\n{'='*70}")
    print(f"  OVERALL WINNER: {winner_name}")
    print(f"  Policy      : {winner_data['metadata']['policy']}")
    print(f"  Mean Reward : {winner_data['mean_reward']:.2f}")
    print(f"  Hyperparams : {winner_data['metadata']['hyperparameters']}")
    print(f"{'='*70}")

    # Copy winning model to project root
    winner_out = Path("best_overall_model")
    winner_out.mkdir(parents=True, exist_ok=True)
    src_model = Path(winner_dir_name) / "dqn_model.zip"
    if src_model.exists():
        shutil.copy2(src_model, winner_out / "dqn_model.zip")
        shutil.copy2(src_model, Path("dqn_model.zip"))
    with (winner_out / "best_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(winner_data["metadata"], f, indent=2)

    # Save full summary
    df_all.to_csv("all_experiments_summary.csv", index=False)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Mathias_Kabango": "#2196F3", "Kellen_Murerwa": "#FF9800", "Edine_Noella_Mugunga": "#4CAF50"}

    ax = axes[0]
    for member in ["Mathias_Kabango", "Kellen_Murerwa", "Edine_Noella_Mugunga"]:
        df_m = df_all[df_all["member"] == member]
        ax.bar(
            [f"{member}\nExp{r['experiment']}" for _, r in df_m.iterrows()],
            df_m["mean_reward"], color=colors[member], edgecolor="black", alpha=0.8,
        )
    ax.set_ylabel("Mean Reward")
    ax.set_title("All 30 Experiments")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    labels = [f"{r['member']}\n({r['policy']})" for _, r in best_per_member.iterrows()]
    means = best_per_member["mean_reward"].tolist()
    stds = best_per_member["std_reward"].tolist()
    bars = ax.bar(labels, means, yerr=stds, capsize=8,
                  color=[colors[m] for m in best_per_member["member"]], edgecolor="black")
    winner_idx = means.index(max(means))
    bars[winner_idx].set_edgecolor("red")
    bars[winner_idx].set_linewidth(3)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Best Model Per Member")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    for i, (_, row) in enumerate(policy_stats.iterrows()):
        ax.bar(row["policy"], row["avg_mean_reward"],
               color=["#9C27B0", "#E91E63"][i], edgecolor="black", alpha=0.85)
        ax.errorbar(row["policy"], row["avg_mean_reward"],
                    yerr=[[row["avg_mean_reward"] - row["worst_mean_reward"]],
                          [row["best_mean_reward"] - row["avg_mean_reward"]]],
                    fmt="none", capsize=10, color="black")
    ax.set_ylabel("Mean Reward")
    ax.set_title("CnnPolicy vs MlpPolicy\n(avg with min/max range)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparison_chart.png", dpi=150)
    plt.show()
    print("\nTraining complete! Best model saved as dqn_model.zip")
