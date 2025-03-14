import os
import sys
import wandb
import yaml  # Import PyYAML

# Configuration variables
PROJECT_NAME = "emg2qwerty"
ENTITY_NAME = "alvister88"
SWEEP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "wandb_sweep_rnn.yaml")

# Set SWEEP_ID here. If you want to use an existing sweep, set it to that ID (e.g., "0i4pz9vg").
# If set to None, a new sweep will be created.
SWEEP_ID = None

NUM_RUNS = 12  # Number of runs to execute in the sweep

def main():
    # Ensure WandB is logged in.
    try:
        wandb.login()
        print("Successfully logged into WandB")
    except Exception as e:
        print(f"Error logging into WandB: {e}")
        print("Please run 'wandb login' first.")
        sys.exit(1)
    
    # Use provided sweep id or create a new one.
    if SWEEP_ID is not None:
        sweep_id = SWEEP_ID
        print(f"Using provided sweep id: {sweep_id}")
    else:
        try:
            print(f"Creating sweep from config: {SWEEP_CONFIG_PATH}")
            with open(SWEEP_CONFIG_PATH, "r") as f:
                sweep_config = yaml.safe_load(f)
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
    
    # Run the sweep agent.
    print(f"Starting sweep agent to run {NUM_RUNS} experiments with sweep id: {sweep_id}")
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
