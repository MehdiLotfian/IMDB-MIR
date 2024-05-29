import sys
import wandb

# Print the current Python executable path
print("Current Python executable:", sys.executable)

# Set the Python executable path for WandB
wandb._executable = sys.executable

# Initialize WandB with your project name
wandb.login(key='ff80b44ece90790b19a7bcb71e054a870e2845c1')
wandb.init(project="your_project")

# Example of logging metrics
for epoch in range(10):
    # Simulate training metrics
    loss = 0.1 * (10 - epoch)
    accuracy = 0.1 * epoch

    # Log metrics to WandB
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})

# Finish the run
wandb.finish()