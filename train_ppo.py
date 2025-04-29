import os
import gymnasium as gym
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from datetime import datetime

# Parse CLI
parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["v1"], default="v1", help="PPO config version")
args = parser.parse_args()

# Output dirs
run_name = f"ppo_{args.config}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
log_dir = f"logs/ppo_breakout/{run_name}"
video_dir = "videos"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Env
def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=False, scale_obs=False)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4, channels_order="last")

# Logger
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Hyperparams
ppo_hyperparams = {
    "learning_rate": 2.5e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "policy_kwargs": dict(net_arch=[512, 256]),
}

# Build model
model = PPO(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    tensorboard_log=log_dir,
    **ppo_hyperparams
)
model.set_logger(new_logger)

# Train
total_timesteps = 1_000_000
model.learn(total_timesteps=total_timesteps)

# Save
model.save(f"ppo_breakout_final_{run_name}")
print(f"✅ Model saved: {run_name}")
