import argparse

from race_track import run_off_policy_monte_carlo


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
    pass

# run off policy monte carlo
elif args.run_off_policy:
    run_off_policy_monte_carlo()

# run both on and off policy monte carlo
elif args.run_on_and_off_policy:
    pass
