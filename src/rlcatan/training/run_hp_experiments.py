import argparse
import subprocess
import sys
import os

def main():
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    search_script = os.path.join(current_folder, "hyperparam_search.py")
    save_script = os.path.join(current_folder, "save_best.py")

    parser = argparse.ArgumentParser(description="Run parallel PPO hyperparameter search.")
    parser.add_argument(
        "--cores", 
        type=int, 
        default=4, 
        help="Number of parallel workers to run (Default: 2)"
    )
    
    args = parser.parse_args()
    num_workers = args.cores

    print(f"--- Starting {num_workers} Parallel Workers ---")
    print(f"Target Script: {search_script}")

    processes = []
    for i in range(num_workers):
        print(f"Launching worker {i+1}...")
        p = subprocess.Popen([sys.executable, search_script])
        processes.append(p)

    print(f"All {num_workers} workers launched! Waiting for them to finish...")

    for p in processes:
        p.wait()

    print("All experiments complete.")
    print("Exporting best results to JSON...")
    subprocess.run([sys.executable, save_script])

if __name__ == "__main__":
    main()