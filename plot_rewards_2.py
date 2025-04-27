import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 📦 Locate the latest run folder
log_dir = "logs/dqn_breakout"
all_runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

if not all_runs:
    raise FileNotFoundError(f"No runs found inside {log_dir}!")

latest_run = max(all_runs, key=os.path.getmtime)
print(f"✅ Using latest run folder: {latest_run}")

# 📦 Find the actual event file
event_file = None
for root, _, files in os.walk(latest_run):
    for file in files:
        if file.startswith("events.out.tfevents"):
            event_file = os.path.join(root, file)
            break

if event_file is None:
    raise FileNotFoundError(f"No TensorBoard event file found inside {latest_run}!")

# 🧹 Load the event data
ea = EventAccumulator(event_file)
ea.Reload()

# 🎯 Tags to plot
tags = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "rollout/exploration_rate",
    "time/fps"
]

# 📂 Create output folder
os.makedirs("plots", exist_ok=True)

# 📈 Plot each tag
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
        filename = os.path.join("plots", tag.replace("/", "_") + ".png")
        plt.savefig(filename)
        print(f"✅ Saved {filename}")
        plt.close()

    except KeyError:
        print(f"❌ Key not found: {tag}")

