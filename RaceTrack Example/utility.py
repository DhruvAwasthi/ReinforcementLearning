import logging
import os
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


def generate_Q():
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} generating Q values for initialization")
    Q_vals = (np.random.rand(26, 13, 6, 6, 9) * 400) - 500
    np.save("data/initialisation/Q_vals.npy", Q_vals)
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} saved Q values for initialization as numpy array")
    return Q_vals


def generate_C():
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} generating C values for initialization")
    C_vals = np.zeros((26, 13, 6, 6, 9), dtype="float64")
    np.save("data/initialisation/C_vals.npy", C_vals)
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} saved C values for initialization as numpy array")
    return C_vals


def generate_policy(Q_vals):
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} generating policy values for initialization")
    policy = np.ones((26, 13, 6, 6), dtype="int64")
    for i in range(26):
        for j in range(13):
            for h in range(6):
                for v in range(6):
                    policy[i, j, h, v] = np.argmax(Q_vals[i, j, h, v, :])
    np.save("data/initialisation/policy.npy", policy)
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} saved policy values for initialization as numpy array")
    return policy


def generate_rewards():
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} generating reward values for initialization")
    rewards = np.array([])
    np.save("data/initialisation/rewards.npy", rewards)
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} saved reward values for initialization as numpy array")
    return rewards


def generate_matrices():
    Q_vals = generate_Q()
    C_vals = generate_C()
    policy = generate_policy(Q_vals)
    rewards = generate_rewards()
    return None


def create_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {dir_path} created successfully")
    except Exception as e:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} exception caused - {e}")
    return

