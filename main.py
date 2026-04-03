from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN, PPO

from environment.custom_env import DroneDeliveryEnv, EnvConfig


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "HOVER",
    5: "RECHARGE",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run best performing trained model")
    p.add_argument("--algo", choices=["auto", "dqn", "pg"], default="auto")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=700)
    return p.parse_args()


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {"mean_reward": float("-inf")}
    return json.loads(path.read_text(encoding="utf-8"))


def choose_algo(requested: str) -> str:
    if requested in ("dqn", "pg"):
        return requested

    dqn_m = _load_metrics(Path("models/dqn/metrics.json"))
    pg_m = _load_metrics(Path("models/pg/metrics.json"))
    return "dqn" if dqn_m.get("mean_reward", -1e9) >= pg_m.get("mean_reward", -1e9) else "pg"


def run(algo: str, episodes: int, max_steps: int) -> None:
    env = DroneDeliveryEnv(EnvConfig(max_steps=max_steps), render_mode="human")

    if algo == "dqn":
        model_path = Path("models/dqn/dqn_final.zip")
        if not model_path.exists():
            model_path = Path("models/dqn/best_model.zip")
        model = DQN.load(str(model_path))
    else:
        model_path = Path("models/pg/pg_final.zip")
        if not model_path.exists():
            model_path = Path("models/pg/best_model.zip")
        model = PPO.load(str(model_path))

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        info = {}
        step_idx = 0

        print(f"[{algo.upper()}] --- episode {ep} start ---")

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_i = int(action)
            obs, reward, terminated, truncated, info = env.step(action_i)
            total_reward += float(reward)
            step_idx += 1

            pickup = info.get("package_position")
            dest = info.get("destination_position")
            events = []
            if info.get("pickups", 0):
                events.append("pickup")
            if info.get("deliveries", 0):
                events.append("delivery")
            if info.get("recharges", 0):
                events.append("recharge")
            if info.get("collisions", 0):
                events.append("collision")
            event_str = ",".join(events) if events else "none"

            print(
                f"[{algo.upper()}][ep {ep:02d} step {step_idx:03d}] "
                f"action={ACTION_NAMES.get(action_i, str(action_i)):<8} "
                f"reward={float(reward):+7.2f} "
                f"battery={info.get('battery', 0.0):6.1f} "
                f"carrying={int(bool(info.get('has_package', False)))} "
                f"pickup={pickup} dest={dest} events={event_str}"
            )

            if env._renderer is not None and not env._renderer.render(info):
                terminated = True

        print(
            f"[{algo.upper()}] episode={ep} reward={total_reward:.2f} "
            f"pickups={info.get('total_pickups', 0)} deliveries={info.get('total_deliveries', 0)} "
            f"recharges={info.get('total_recharges', 0)} collisions={info.get('total_collisions', 0)}"
        )

    env.close()


if __name__ == "__main__":
    args = parse_args()
    selected = choose_algo(args.algo)
    print(f"Running model: {selected}")
    run(selected, args.episodes, args.max_steps)
