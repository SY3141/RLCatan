import optuna
from optuna.samplers import RandomSampler
import os
import time
import datetime
from stable_baselines3.common.evaluation import evaluate_policy

from train_ppo_v1 import train_ppo, make_env

STUDY_NAME = "ppo_parallel_search"
STORAGE_URL = "sqlite:///catan_hyperparams.db"
TRAINING_STEPS = 1_000_000
EVAL_EPISODES = 50

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    n_epochs = trial.suggest_categorical("n_epochs", [3, 4, 5, 10])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = (
        f"[{timestamp}] Trial {trial.number} STARTED: "
        f"LR={learning_rate:.5f}, Batch={batch_size}, Steps={n_steps}, "
        f"Epochs={n_epochs}, Gamma={gamma}, Ent={ent_coef:.4f}\n"
    )
    
    print(log_message.strip())
    try:
        with open("running_log.txt", "a") as f:
            f.write(log_message)
    except:
        pass


    try:
        model = train_ppo(
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

        # evaluate the trained model
        eval_env = make_env()
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES) 
        
        result_msg = f"Trial {trial.number} FINISHED. Score: {mean_reward:.2f} (+/- {std_reward:.2f})\n"
        print(result_msg.strip())
        
        with open("running_log.txt", "a") as f:
            f.write(result_msg)
        return mean_reward

    except Exception as e:
        print(f"Trial {trial.number} FAILED: {e}")
        return -100.0

if __name__ == "__main__":
    os.makedirs("optuna_logs", exist_ok=True)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        load_if_exists=True,
        sampler=RandomSampler()
    )

    print("Worker started")
    study.optimize(objective, n_trials=1) # Each worker runs one trial at a time, for good cpu's/gpu's utilization make n_trials larger