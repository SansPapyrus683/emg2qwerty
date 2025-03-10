import os
import sys
import wandb

# Fixed configuration for your sweep
SWEEP_CONFIG_PATH = "wandb_sweep_config.yaml"
PROJECT_NAME = "emg2qwerty"
ENTITY_NAME = "alvister88"
NUM_RUNS = 20  # Number of runs to execute in the sweep

def main():
    # Ensure WandB is logged in
    try:
        wandb.login()
        print("Successfully logged into WandB")
    except Exception as e:
        print(f"Error logging into WandB: {e}")
        print("Please run 'wandb login' first.")
        sys.exit(1)
    
    # Initialize the sweep
    try:
        print(f"Creating sweep from config: {SWEEP_CONFIG_PATH}")
        with open(SWEEP_CONFIG_PATH, "r") as f:
            sweep_config = wandb.yaml.load(f.read())
            
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=PROJECT_NAME,
            entity=ENTITY_NAME
        )
        print(f"Sweep created with ID: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/{ENTITY_NAME}/{PROJECT_NAME}/sweeps/{sweep_id}")
    except Exception as e:
        print(f"Error creating sweep: {e}")
        sys.exit(1)
    
    # Run the sweep agent
    print(f"Starting sweep agent to run {NUM_RUNS} experiments")
    try:
        wandb.agent(
            sweep_id=sweep_id,
            project=PROJECT_NAME,
            entity=ENTITY_NAME,
            count=NUM_RUNS
        )
    except Exception as e:
        print(f"Error running sweep agent: {e}")
        sys.exit(1)
    
    print("Sweep completed successfully!")

if __name__ == "__main__":
    main()