from __future__ import annotations

import argparse
import numpy as np

from env import DroneDeliveryEnv, EnvConfig
from dqn import DQNAgent
from ppo import PPOAgent
from reinforce import ReinforceAgent


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "HOVER",
    5: "RECHARGE",
}


def evaluate(
    algo: str,
    model_path: str,
    episodes: int,
    max_steps: int,
    render: bool,
    show_interaction: bool,
) -> None:
    env = DroneDeliveryEnv(EnvConfig(max_steps=max_steps), seed=123)

    if algo == "dqn":
        agent = DQNAgent(env.obs_dim, env.n_actions)
    elif algo == "ppo":
        agent = PPOAgent(env.obs_dim, env.n_actions)
    else:
        agent = ReinforceAgent(env.obs_dim, env.n_actions)

    agent.load(model_path)

    vis = None
    if render:
        from visualization import CityVisualization

        vis = CityVisualization(env)

    ep_rewards = []
    ep_pickups = []
    ep_deliveries = []
    ep_collisions = []
    ep_recharges = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        info = {}
        step_idx = 0

        if show_interaction:
            print(f"[{algo.upper()}] --- episode {ep} start ---")

        while not done:
            action = int(agent.act_greedy(obs[0]))
            actions = [action]
            obs, reward, done, _, info = env.step(actions)
            total_reward += reward
            step_idx += 1

            if show_interaction:
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
                    f"action={ACTION_NAMES.get(action, str(action)):<8} "
                    f"reward={reward:+7.2f} "
                    f"battery={info.get('battery', 0.0):6.1f} "
                    f"carrying={int(bool(info.get('has_package', False)))} "
                    f"pickup={pickup} dest={dest} events={event_str}"
                )

            if vis is not None:
                if not vis.handle_events():
                    done = True
                vis.render(reward, info)

        ep_rewards.append(total_reward)
        ep_pickups.append(info.get("total_pickups", 0))
        ep_deliveries.append(info.get("total_deliveries", 0))
        ep_collisions.append(info.get("total_collisions", 0))
        ep_recharges.append(info.get("total_recharges", 0))
        print(
            f"[{algo.upper()}] eval ep={ep:3d} reward={total_reward:8.2f} "
            f"pickups={ep_pickups[-1]:3d} deliveries={ep_deliveries[-1]:3d} "
            f"recharges={ep_recharges[-1]:3d} collisions={ep_collisions[-1]:3d}"
        )

    print("---")
    print(f"{algo.upper()} mean reward: {float(np.mean(ep_rewards)):.2f}")
    print(f"{algo.upper()} mean pickups: {float(np.mean(ep_pickups)):.2f}")
    print(f"{algo.upper()} mean deliveries: {float(np.mean(ep_deliveries)):.2f}")
    print(f"{algo.upper()} mean recharges: {float(np.mean(ep_recharges)):.2f}")
    print(f"{algo.upper()} mean collisions: {float(np.mean(ep_collisions)):.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run trained drone delivery agents")
    p.add_argument("--algo", choices=["dqn", "ppo", "reinforce"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=700)
    p.add_argument("--render", action="store_true")
    p.add_argument("--show-interaction", action="store_true", help="Print step-by-step agent/environment interaction")
    args = p.parse_args()

    evaluate(
        args.algo,
        args.model,
        args.episodes,
        args.max_steps,
        args.render,
        args.show_interaction,
    )


if __name__ == "__main__":
    main()
