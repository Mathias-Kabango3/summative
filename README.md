# Drone Delivery RL (Gymnasium + SB3)

Single-agent drone delivery environment on a clean grid. The drone must pick up a package, deliver it to a destination, avoid obstacles, and recharge at charging stations.

## Required Project Structure

```text
project_root/
├── environment/
│   ├── custom_env.py
│   └── rendering.py
├── training/
│   ├── dqn_training.py
│   └── pg_training.py
├── models/
│   ├── dqn/
│   └── pg/
├── main.py
├── requirements.txt
└── README.md
```

## Environment Highlights

- Grid-only map with a clean visual layout
- Multiple charging stations with battery symbols (and tiny CHG labels)
- Obstacle cells the drone must avoid
- Explicit package marker (tiny PKG label)
- Explicit destination marker (tiny DEST label)
- Battery-aware reward shaping to encourage smart charging behavior

## Action Space

- 0: up
- 1: down
- 2: left
- 3: right
- 4: hover
- 5: recharge (valid at charging stations)

## Install

```bash
python -m pip install -r requirements.txt
```

## Train (about 2M timesteps)

Train DQN:

```bash
python training/dqn_training.py --timesteps 2000000 --output models/dqn
```

Train PPO (policy gradient):

```bash
python training/pg_training.py --timesteps 2000000 --output models/pg
```

Train REINFORCE:

```bash
python train.py --algo reinforce --episodes 2000 --out-dir models/reinforce
```

Each training script writes:

- final model zip
- best model checkpoint(s) from evaluation callback
- metrics.json with mean reward

The shared `train.py` entry point also supports `dqn` and `ppo` if you want to run the same episodic trainer for those agents.

## Run Best Performing Model

```bash
python main.py --algo auto --episodes 3 --max-steps 700
```

You can also force a specific model:

```bash
python main.py --algo dqn --episodes 3
python main.py --algo pg --episodes 3
```
