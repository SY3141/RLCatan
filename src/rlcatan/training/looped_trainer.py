import argparse
import time
from continue_train_ppo import ppo_train


def run_many(num_runs, num_iterations, model_name):
    """
    Executes the ppo_train function multiple times based on provided arguments.
    """
    start_time = time.time()

    for i in range(1, num_runs + 1):
        print(f"\n==========================")
        print(f"  Starting training run {i}/{num_runs}")
        print(f"  Iterations: {num_iterations}")
        print(f"==========================\n")

        # Call the training function with the dynamic iteration count
        ppo_train(num_iterations, model_name)

    total_time = time.time() - start_time
    print(f"\nAll {num_runs} runs completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PPO training multiple times with custom iterations."
    )

    parser.add_argument(
        "-runs",
        "--runs",
        type=int,
        default=10,
        help="Number of times to execute the training loop (default: 10)",
    )

    parser.add_argument(
        "-iterations",
        "--iterations",
        "-iter",
        "--iter",
        type=int,
        default=1_000_000,
        help="Number of iterations to pass to ppo_train (default: 1,000,000)",
    )

    parser.add_argument(
        "-name",
        "--name",
        type=str,
        default="ppo_v4",
        help="Model name to pass to ppo_train (default: ppo_v4)",
    )

    args = parser.parse_args()
    run_many(args.runs, args.iterations, args.name)
