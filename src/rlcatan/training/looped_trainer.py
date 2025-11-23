from continue_train_ppo import ppo_train   # or your actual filename
import time

NUM_RUNS = 10       # change this to however many cycles you want

def run_many():
    for i in range(1, NUM_RUNS + 1):
        print(f"\n==========================")
        print(f"  Starting training run {i}/{NUM_RUNS}")
        print(f"==========================\n")

        ppo_train(1_000) 


if __name__ == "__main__":
    run_many()
