# hyperparams.py

DQN_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.985,
    "buffer_size": 200_000,
    "learning_starts": 10_000,
    "batch_size": 64,
    "tau": 1.0,
    "train_freq": 4,
    "target_update_interval": 20_000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.02,
    "policy_kwargs": dict(
        net_arch=[256, 256],
        normalize_images=False,
    ),
}
