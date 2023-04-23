import argparse
import logging
import os
from datetime import datetime

from config import LOG_DIR, SAVE_RESULTS_DIR
from race_track import run_off_policy_monte_carlo, run_on_policy_monte_carlo
from utility import create_dir


# create directories
create_dir(SAVE_RESULTS_DIR)
create_dir(LOG_DIR)

logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(LOG_DIR,
                                              datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"))
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Racetrack Example - Reinforcement Learning",
    epilog="If you like the work, please star the repository https://github.com/DhruvAwasthi/ReinforcementLearning")


parser.add_argument("--run_on_policy",
                    default=False,
                    type=bool,
                    help="set this to true to run on policy monte carlo")

parser.add_argument("--run_off_policy",
                    default=False,
                    type=bool,
                    help="set this to true to run off policy monte carlo")

parser.add_argument("--run_on_and_off_policy",
                    default=False,
                    type=bool,
                    help="set this true to run both on and off policy monte carlo")

args = parser.parse_args()


# run on policy monte carlo
if args.run_on_policy:
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} running on policy monte carlo")
    run_on_policy_monte_carlo()
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully executed on policy monte carlo")

# run off policy monte carlo
elif args.run_off_policy:
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} running off policy monte carlo")
    run_off_policy_monte_carlo()
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully executed off policy monte carlo")

# run both on and off policy monte carlo
elif args.run_on_and_off_policy:
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} running on policy monte carlo")
    run_on_policy_monte_carlo()
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully executed on policy monte carlo")

    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} running off policy monte carlo")
    run_off_policy_monte_carlo()
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully executed off policy monte carlo")
