import optuna
from optuna.samplers import RandomSampler
import os
import time
import datetime
import sys
from train_ppo_v1 import train_ppo

# Configuration
STUDY_NAME = "ppo_parallel_search"
STORAGE_URL = "sqlite:///catan_hyperparams.db"
TRAINING_STEPS = 1_000_000 
N_TRIALS_PER_WORKER = 4  # Each worker will attempt this many trials
LOG_FILE = "all_trials_log.txt"
BEST_PARAMS_FILE = "best_params_log.txt"

class FileLock:
    """
    A simple cross-platform file lock using directory creation (atomic).
    This prevents multiple workers from writing to the log file simultaneously.
    """
    def __init__(self, lock_name):
        self.lock_name = lock_name + ".lock"

    def __enter__(self):
        timeout = 10  # seconds
        start_time = time.time()
        while True:
            try:
                os.mkdir(self.lock_name)
                return
            except FileExistsError:
                if time.time() - start_time > timeout:
                    # Break lock if it's stale (process crashed holding lock)
                    print(f"Breaking stale lock: {self.lock_name}")
                    try:
                        os.rmdir(self.lock_name)
                    except:
                        pass
                time.sleep(0.05)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.rmdir(self.lock_name)
        except:
            pass

def save_log(filename, message):
    """Safely appends a message to a file using the lock."""
    with FileLock(filename):
        with open(filename, "a") as f:
            f.write(message + "\n")

def objective(trial):
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    n_epochs = trial.suggest_categorical("n_epochs", [3, 4, 5, 10])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worker_pid = os.getpid()
    
    start_msg = (
        f"[{timestamp}] [PID:{worker_pid}] Trial {trial.number} STARTED: "
        f"LR={learning_rate:.5f}, Batch={batch_size}, Steps={n_steps}, "
        f"Epochs={n_epochs}, Gamma={gamma}, Ent={ent_coef:.4f}"
    )
    print(start_msg)
    save_log(LOG_FILE, start_msg)

    try:
        # 2. Train and Evaluate
        # train_ppo returns the ep_rew_mean (float)
        score = train_ppo(
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=n_epochs,      
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            total_timesteps=TRAINING_STEPS,
        )
        
        # 3. Log Result
        end_msg = f"[{timestamp}] [PID:{worker_pid}] Trial {trial.number} FINISHED. Score: {score:.2f}"
        print(end_msg)
        save_log(LOG_FILE, end_msg)
        
        # 4. Check and Log Best (Post-Trial)
        # We query the study to see if this trial is the new best
        try:
            study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
            if study.best_trial.number == trial.number:
                best_msg = (
                    f"\n>>> NEW BEST FOUND by PID:{worker_pid} <<<\n"
                    f"Trial: {trial.number}\n"
                    f"Score: {score:.2f}\n"
                    f"Params: {trial.params}\n"
                    f"Timestamp: {timestamp}\n"
                    f"---------------------------------------------"
                )
                print(best_msg)
                save_log(BEST_PARAMS_FILE, best_msg)
        except Exception as e:
            # Sometimes accessing best_trial fails if it's the very first trial
            pass

        return score

    except Exception as e:
        fail_msg = f"[{timestamp}] [PID:{worker_pid}] Trial {trial.number} FAILED: {str(e)}"
        print(fail_msg)
        save_log(LOG_FILE, fail_msg)
        return -100.0

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("optuna_logs", exist_ok=True)
    
    # Retry connecting to DB (SQLite can be locked briefly by other workers)
    for _ in range(5):
        try:
            study = optuna.create_study(
                study_name=STUDY_NAME,
                storage=STORAGE_URL,
                direction="maximize",
                load_if_exists=True,
                sampler=RandomSampler()
            )
            break
        except Exception as e:
            print(f"Database locked, retrying... {e}")
            time.sleep(1)

    print(f"Worker {os.getpid()} started. Running {N_TRIALS_PER_WORKER} trials.")
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS_PER_WORKER)