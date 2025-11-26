import argparse
import subprocess
import sys
import os
import time

def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    search_script = os.path.join(current_folder, "hyperparam_search.py")

    
    if not os.path.exists(search_script):
        print(f"\nCRITICAL ERROR: Could not find worker script!")
        print(f"Looked at: {search_script}")
        print("Make sure 'hyperparam_search.py' is in the same folder as this script.\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run parallel PPO hyperparameter search with Optuna.")
    
    parser.add_argument(
        "--cores", 
        type=int, 
        default=1,  # Default to 1 core if not specified
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
        # Removed the env=env part since we aren't passing IDs anymore
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

    #  Print Best Results Summary at End
    print("\n==================================================")
    print("   EXPERIMENT FINISHED")
    print("==================================================")
    
    try:
        import optuna
        # Connect to DB to fetch the best result
        study = optuna.load_study(
            study_name="ppo_parallel_search", 
            storage="sqlite:///catan_hyperparams.db"
        )
        print(f"Best Trial Found (All-Time):")
        print(f"  Trial Number: {study.best_trial.number}")
        print(f"  Best Score:   {study.best_trial.value:.4f}")
        print(f"  Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    except ImportError:
        print("Could not print summary: 'optuna' library not found in this environment.")
    except Exception as e:
        print(f"Could not load study summary: {e}")
    print("==================================================\n")

if __name__ == "__main__":
    main()