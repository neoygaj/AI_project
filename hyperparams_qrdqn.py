# hyperparams_qrdqn.py

QRDQN_HYPERPARAMS = {
    "learning_rate": 1e-4,        # same idea, maybe bump later
    "gamma": 0.99,                # higher gamma often good for QRDQN
    "buffer_size": 500_000,
    "learning_starts": 10_000,
    "batch_size": 64,             # larger batch often helps (64 or even 128)
    "tau": 1.0,
    "train_freq": 4,
    "target_update_interval": 15_000,  # 🔥 maybe less frequent target updates
    "exploration_fraction": 0.05,
    "exploration_final_eps": 0.05,
                
    "policy_kwargs": dict(
        net_arch=[1024, 512, 256],       # hidden layers
        normalize_images=False,         # 🔥 always False for Atari
        n_quantiles=51,   
    ),
}
