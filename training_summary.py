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
tags = {
    "rollout/ep_rew_mean": "Mean Episode Reward",
    "rollout/ep_len_mean": "Mean Episode Length",
    "rollout/exploration_rate": "Exploration Rate",
    "time/fps": "Frames Per Second"
}

# 📜 Output PDF
output_pdf = "training_summary.pdf"

# 📈 Plot everything
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, (tag, title) in enumerate(tags.items()):
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        axs[i].plot(steps, values)
        axs[i].set_title(title)
        axs[i].set_xlabel("Timesteps")
        axs[i].set_ylabel(title)
        axs[i].grid(True)

    except KeyError:
        axs[i].text(0.5, 0.5, f"Missing: {tag}", ha='center', va='center')
        axs[i].set_title(title)
        axs[i].axis('off')

plt.tight_layout()
plt.savefig(output_pdf)
print(f"✅ Saved {output_pdf}")

