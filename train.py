from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dqn import DQNAgent
from ppo import PPOAgent
from reinforce import ReinforceAgent
from environment.custom_env import DroneDeliveryEnv, EnvConfig
from utils import CsvLogger, EpisodeLog


def train(
    env,
    algo: Literal["dqn", "ppo", "reinforce"],
    episodes: int,
    render: bool = False,
    checkpoint_every: int = 25,
    out_dir: str = "models",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if algo == "dqn":
        agent: Any = DQNAgent(env.obs_dim, env.n_actions)
    elif algo == "ppo":
        agent = PPOAgent(env.obs_dim, env.n_actions)
    else:
        agent = ReinforceAgent(env.obs_dim, env.n_actions)

    logger = CsvLogger(out / f"{algo}_train_log.csv")
    rewards = []

    vis = None
    if render:
        from visualization import CityVisualization

        vis = CityVisualization(env)

    for ep in range(1, episodes + 1):
        obs_list, _ = env.reset()
        ep_reward = 0.0
        info = {}

        for _ in range(env.cfg.max_steps):
            actions = [agent.act(obs) for obs in obs_list]
            next_obs, reward, done, _, info = env.step(actions)

            # Shared policy updated from each drone transition.
            per_drone = info["per_drone_rewards"]
            for i in range(env.cfg.n_drones):
                if algo == "dqn":
                    agent.remember(obs_list[i], actions[i], per_drone[i], next_obs[i], done)
                elif algo == "ppo":
                    agent.store_outcome(per_drone[i], done)
                else:
                    agent.store_reward(per_drone[i])

            if algo == "dqn":
                agent.update()

            obs_list = next_obs
            ep_reward += reward

            if vis is not None:
                if not vis.handle_events():
                    done = True
                vis.render(reward, info)

            if done:
                break

        if algo in ("ppo", "reinforce"):
            agent.update()

        rewards.append(ep_reward)
        logger.append(
            EpisodeLog(
                episode=ep,
                reward=ep_reward,
                deliveries=info.get("total_deliveries", 0),
                collisions=info.get("total_collisions", 0),
            )
        )

        mean_last = float(np.mean(rewards[-20:]))
        print(
            f"[{algo.upper()}] ep={ep:4d} reward={ep_reward:8.2f} "
            f"mean20={mean_last:8.2f} deliveries={info.get('total_deliveries', 0):3d} "
            f"collisions={info.get('total_collisions', 0):3d}"
        )

        if ep % checkpoint_every == 0:
            agent.save(str(out / f"{algo}_ep_{ep}.pt"))

    agent.save(str(out / f"{algo}_final.pt"))
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drone delivery agents")
    parser.add_argument("--algo", choices=["dqn", "ppo", "reinforce"], required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--max-steps", type=int, default=700)
    args = parser.parse_args()

    env = DroneDeliveryEnv(EnvConfig(max_steps=args.max_steps))
    train(
        env=env,
        algo=args.algo,
        episodes=args.episodes,
        render=args.render,
        checkpoint_every=args.checkpoint_every,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
