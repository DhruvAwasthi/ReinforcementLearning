import numpy as np


def generate_Q():
    Q_vals = (np.random.rand(26, 13, 6, 6, 9) * 400) - 500
    np.save("data/Q_vals.npy", Q_vals)
    return Q_vals


def generate_C():
    C_vals = np.zeros((26, 13, 6, 6, 9), dtype="float64")
    np.save("data/C_vals.npy", C_vals)
    return C_vals


def generate_policy(Q_vals):
    policy = np.ones((26, 13, 6, 6), dtype="int64")
    for i in range(26):
        for j in range(13):
            for h in range(6):
                for v in range(6):
                    policy[i, j, h, v] = np.argmax(Q_vals[i, j, h, v, :])
    np.save("data/policy.npy", policy)
    return policy


def generate_rewards():
    rewards = np.array([])
    np.save("data/rewards.npy", rewards)
    return rewards


def generate_matrices():
    Q_vals = generate_Q()
    C_vals = generate_C()
    policy = generate_policy(Q_vals)
    rewards = generate_rewards()
    return None


generate_matrices()
