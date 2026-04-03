from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN

from environment.custom_env import DroneDeliveryEnv, EnvConfig


def resolve_model_path(model_dir: Path) -> Path:
    primary = model_dir / "dqn_final.zip"
    fallback = model_dir / "best_model.zip"

    if primary.exists():
        return primary
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"No DQN model found in {model_dir}. Expected dqn_final.zip or best_model.zip"
    )


def run(episodes: int = 3, max_steps: int = 700) -> None:
    model_path = resolve_model_path(Path("models/dqn"))
    model = DQN.load(str(model_path))
    env = DroneDeliveryEnv(EnvConfig(max_steps=max_steps), render_mode="human")

    print(f"Loaded DQN model: {model_path}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)

            if env._renderer is not None and not env._renderer.render(info):
                terminated = True

        print(
            f"[DQN] episode={ep} reward={total_reward:.2f} "
            f"deliveries={info.get('total_deliveries', 0)} "
            f"collisions={info.get('total_collisions', 0)}"
        )

    env.close()


if __name__ == "__main__":
    run(episodes=3, max_steps=700)
