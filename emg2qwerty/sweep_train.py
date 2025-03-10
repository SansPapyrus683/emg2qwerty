import yaml
import wandb
import subprocess

def load_sweep_config(config_path="sweep_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_train():
    # Initialize a new wandb run for the current sweep trial.
    run = wandb.init()
    config = wandb.config

    # Build a list of Hydra command-line overrides from the wandb config.
    overrides = []
    if "seed" in config:
        overrides.append(f"seed={config['seed']}")
    if "optimizer.lr" in config:
        overrides.append(f"optimizer.lr={config['optimizer.lr']}")

    # Join overrides into a single string.
    override_str = " ".join(overrides)

    # Run the original training script with the overrides.
    # The original training function is in `train.py` (its main function decorated by Hydra)
    command = f"python train.py {override_str}"
    subprocess.run(command, shell=True)

    run.finish()

def sweep_train():
    # Load sweep configuration from YAML file.
    sweep_configuration = load_sweep_config("sweep_config.yaml")
    # Create a new sweep in wandb.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="emg2qwerty-sweep")
    # Launch the agent to run the training multiple times.
    wandb.agent(sweep_id, function=run_train, count=10)

if __name__ == "__main__":
    sweep_train()
