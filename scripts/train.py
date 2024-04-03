import wandb
import random

# Set seed for reproducibility
random.seed(10)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="player-segmentation",
    # set the wandb entity where this run will be logged
    entity="ludekcizinsky",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    
