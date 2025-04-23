import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "logs/dqn_breakout/DQN_1"

ea = EventAccumulator(log_dir)
ea.Reload()

tags = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "rollout/exploration_rate",
    "time/fps"
]

for tag in tags:
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure()
        plt.plot(steps, values)
        plt.xlabel("Timesteps")
        plt.ylabel(tag.split("/")[-1].replace("_", " ").capitalize())
        plt.title(tag)
        plt.grid(True)
        filename = tag.replace("/", "_") + ".png"
        plt.savefig(filename)
        print(f"✅ Saved {filename}")
        plt.close()

    except KeyError:
        print(f"❌ Key not found: {tag}")
