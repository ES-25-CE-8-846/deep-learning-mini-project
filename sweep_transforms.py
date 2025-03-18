from single_gpu import Trainer, Tester
from fruits_and_veggies_dataloader_simple import FruitsAndVeggies, FruitsAndVeggiesAugmentator 
import wandb 
import torch 
import torchvision
import os 
from torch.utils.data import DataLoader


with open("/home/ai/.wandb.key", 'r') as wandb_key_file:
    wandb.login(key=wandb_key_file.read().strip(), relogin=True)


path_to_data = "/home/ai/datasets/kaggel/fruits_and_veggies/8/"

sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "test_macc"},
    "parameters": {
        "augmentations": {"values": ['all', 'ColorJitter', 'RandomHorizontalFlip', 'RandomErasing', 'RandomRotation','none']},
    },
}

sweep_id = wandb.sweep(sweep_config, project="test-Mini-project")


def main():
    print("Main entered")
    learning_rate = 0.0001
    learning_rate_scheduler = "non"
    epochs = 10
    batch_size = 32
    model_name = "vit_b_16"
    n_layers_to_freeze = 0
    pretrained_weights = "DEFAULT"
    log_to_wandb = True
    with_augs = False

    run = wandb.init(name=f"a_bs{batch_size}_{model_name}_augs",
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="avs-846",
        # Set the wandb project where this run will be logged.
        project="test-Mini-project",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "architecture": model_name,
            "dataset": "fruit-and-vegetable-image-recognition",
            "epochs": epochs,
            "batch_size":batch_size,
            "n_frosen_layers":n_layers_to_freeze,
            "pretrained_weights":pretrained_weights,
        },
    )
    
    augments_to_use = wandb.config.augmentations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available

    print(f"Using device: {device}")

    # Initialize the model and move it to the GPU if available
    model = torchvision.models.vit_b_16(weights=pretrained_weights).to(device)
    num_features = model.heads.head.in_features  # Get the input features of the last layer
    model.heads.head = torch.nn.Linear(num_features, 36).to(device)  # Replace it with a new layer


    model_transforms = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()

    print("Creating dataloaders...")
    if with_augs:
        train_augs = FruitsAndVeggiesAugmentator(augments_to_use).augementation 
    else:
        train_augs = None


    train_dataset = FruitsAndVeggies(os.path.join(path_to_data, "train"), model_transforms, train_augs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=32)
    
    validation_dataset = FruitsAndVeggies(os.path.join(path_to_data, "validation"), model_transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, pin_memory=True)

    test_dataset = FruitsAndVeggies(os.path.join(path_to_data, "test"), model_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(model = model,
                      train_dataloader = train_dataloader,
                      validation_dataloader = validation_dataloader,
                      optimizer = optimizer,
                      device = device,
                      model_name = model_name,
                      experiment_path = f"a_bs{batch_size}_{model_name}_freeze_{n_layers_to_freeze}")

    tester = Tester(model=model, 
                    test_dataloader = test_dataloader,
                    dataset = test_dataset,
                    device = device,
                    experiment_path = f"a_bs{batch_size}_{model_name}_freeze_{n_layers_to_freeze}")

    for epoch in range(epochs):  # Run for 10 epochs
        trainer.run_epoch(epoch)

    tester.run_test()



wandb.agent(sweep_id, function=main, count=4)

