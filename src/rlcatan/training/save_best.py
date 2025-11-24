import optuna
import json

STUDY_NAME = "ppo_parallel_search"
STORAGE_URL = "sqlite:///catan_hyperparams.db"
JSON_OUTPUT = "best_hyperparams.json"
TXT_OUTPUT = "complete_history.txt"

def main():
    print(f"Loading results from {STORAGE_URL}...")
    
    study = optuna.load_study(
        study_name=STUDY_NAME, 
        storage=STORAGE_URL
    )

    print(f"Total Trials found: {len(study.trials)}")
    
    if len(study.trials) == 0:
        print("No trials found. Exiting.")
        return

    # 1. Save Winner to JSON (Machine Readable)
    print(f"Best Score: {study.best_value}")
    results = {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "n_trials": len(study.trials)
    }
    with open(JSON_OUTPUT, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved winner to {JSON_OUTPUT}")

    # Sort trials by value (Score), highest first. Filter out failures
    completed_trials = [t for t in study.trials if t.value is not None]
    completed_trials.sort(key=lambda t: t.value, reverse=True)

    with open(TXT_OUTPUT, "w") as f:
        f.write(f"Experiment History - {len(completed_trials)} trials completed\n")
        f.write("=================================================\n")
        
        for i, trial in enumerate(completed_trials):
            f.write(f"Rank {i+1}: Trial {trial.number}\n")
            f.write(f"  Score: {trial.value:.4f}\n")
            f.write(f"  Params: {trial.params}\n")
            f.write("-------------------------------------------------\n")
            
    print(f"Saved full history log to {TXT_OUTPUT}")

if __name__ == "__main__":
    main()