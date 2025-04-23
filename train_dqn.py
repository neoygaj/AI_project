import os
import gymnasium as gym
import numpy as np
import ale_py  # 🔑 registers ALE environments
from gymnasium.spaces import Box
from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from datetime import datetime


# Output directories
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/dqn_breakout/{run_name}"
checkpoint_dir = "checkpoints/longrun"
video_dir = "videos"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Environment factory
def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)  # ✅ Correct: directly passed
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=False, scale_obs=False, screen_size=84, terminal_on_life_loss=False, noop_max=30)
    env = Monitor(env)
    return env


# Vectorized environment with frame stacking
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4, channels_order="last")

print("🚨 Obs shape:", env.observation_space)


# Save checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=checkpoint_dir,
    name_prefix="dqn_breakout"
)

# DQN model
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=dict(normalize_images=False),
)
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# Train for 200k steps
model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
model.save("dqn_breakout_final")

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"✅ Evaluation: mean_reward={mean_reward:.2f}, std={std_reward:.2f}")

name_prefix = f"dqn-breakout-{datetime.now().strftime('%Y%m%d-%H%M%S')}" # adds a unique UNIX timestamp

# 🎥 Record a demo video
record_env = DummyVecEnv([make_env])
record_env = VecFrameStack(record_env, n_stack=4)
record_env = VecVideoRecorder(
    record_env,
    video_folder=video_dir,
    record_video_trigger=lambda step: step == 0,
    video_length=1000,
    name_prefix=name_prefix
)

obs = record_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = record_env.step(action)

record_env.close()
print(f"🎥 Video saved to: {video_dir}")