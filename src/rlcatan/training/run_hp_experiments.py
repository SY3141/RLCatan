import argparse
import subprocess
import sys
import os
import time

def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    search_script = os.path.join(current_folder, "hyperparam_search.py")

    parser = argparse.ArgumentParser(description="Run parallel PPO hyperparameter search with Optuna.")
    
    parser.add_argument(
        "--cores", 
        type=int, 
        default=26, 
        help="Number of parallel workers to launch."
    )
    
    args = parser.parse_args()
    num_workers = args.cores

    print(f"==================================================")
    print(f"   Starting Optuna Hyperparameter Search")
    print(f"   Workers: {num_workers}")
    print(f"   Script: {search_script}")
    print(f"   Storage: sqlite:///catan_hyperparams.db")
    print(f"   Logs: all_trials_log.txt, best_params_log.txt")
    print(f"==================================================\n")

    # Clear previous locks if any exist from a hard crash
    if os.path.exists("all_trials_log.txt.lock"):
        try:
            os.rmdir("all_trials_log.txt.lock")
            print("Cleared stale lock file.")
        except:
            pass

    processes = []
    for i in range(num_workers):
        print(f"Launching worker {i+1}/{num_workers}...")
        # Use sys.executable to ensure we use the same python interpreter
        p = subprocess.Popen([sys.executable, search_script])
        processes.append(p)
        # Stagger start times slightly to reduce initial DB lock contention
        time.sleep(1)

    print(f"\nAll {num_workers} workers are running.")
    print("Press Ctrl+C to stop all workers manually.")

    try:
        # Wait for all processes to finish
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all workers...")
        for p in processes:
            p.terminate()
        print("Done.")

if __name__ == "__main__":
    main()