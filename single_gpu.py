import torch
from torch.utils.data import DataLoader
import torchvision
from fruits_and_veggies_dataloader_simple import FruitsAndVeggies
import os
from tqdm import tqdm
import wandb



with open("/ceph/home/student.aau.dk/bd45mb/wandb.txt", 'r') as wandb_key_file:
    wandb.login(key=wandb_key_file.read().strip(), relogin=True)



path_to_data = "/ceph/home/student.aau.dk/bd45mb/datasets/kaggle_fruits_and_veggies/"

class Trainer:
    def __init__(self, model, train_dataloader, validation_dataloader, optimizer: torch.optim.Optimizer, device, model_name):
        self.model = model.to(device)  # Move model to the device (GPU)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.current_epoch = 0
        self.model_name = model_name

    def _run_batch(self, data: torch.Tensor, labels: torch.Tensor):
        data, labels = data.to(self.device), labels.to(self.device)  # Move data and labels to GPU
        self.optimizer.zero_grad()
        pred = self.model(data)
        loss = self.loss_function(pred, labels)
        loss.backward()
        print(loss.item())
        self.optimizer.step()
        self.last_val_loss = 0
        self.best_val_loss = 100
        wandb.log({"train_loss":loss.item(),
                    "train_acc":self._compute_acc(pred, labels)})
        return loss.item()

    def _run_validation_batch(self, data: torch.Tensor, labels: torch.Tensor):
        data, labels = data.to(self.device), labels.to(self.device)  # Move data and labels to GPU
        pred = self.model(data)
        loss = self.loss_function(pred, labels)
        return loss.item()

    def _run_epoch(self, epoch):
        self.current_epoch = epoch
        self.model.train()
        b_sz = len(next(iter(self.train_dataloader))[0])
        print(f"Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_dataloader)}")
        epoch_loss = 0
        for data, labels in self.train_dataloader:
            batch_loss = self._run_batch(data, labels)

            epoch_loss += batch_loss

        epoch_loss = epoch_loss/len(self.train_dataloader)
        wandb.log({"train_loss_epoch":epoch_loss})


        self.model.eval()
        validation_loss = 0
        for data, labels in self.validation_dataloader:
            batch_loss = self._run_batch(data, labels)
            validation_loss += batch_loss
        validation_loss = validation_loss / len(self.validation_dataloader)
        wandb.log({"val_loss":validation_loss})

        if validation_loss < self.best_val_loss:
            self.best_val_loss = validation_loss
        self.last_val_loss = validation_loss

        self._save_checkpoint()

    def _compute_acc(self, preds, labels,):
        preds_index = torch.argmax(preds, dim=1)
        acc = labels == preds_index
        acc = torch.sum(acc)/acc.shape[0]
        return acc

    def _save_checkpoint(self):
        path = "checkpoints"
        os.makedirs(path, exist_ok=True)
        file_name = f"{self.current_epoch}_".rjust(5,"0") + f"{self.model_name}_{round(self.last_val_loss, 3)}__{round(self.best_val_loss, 3)}.pth"

        full_save_path = os.path.join(path, file_name)
        torch.save(self.model.state_dict(), full_save_path)



def main():
    print("Main entered")
    learning_rate = 0.0001
    epochs = 50
    batch_size = 128
    model_name = "vit_b_32"
    run = wandb.init(name="test2",
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
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    print(f"Using device: {device}")

    # Initialize the model and move it to the GPU if available
    model = torchvision.models.vit_b_32(weights="IMAGENET1K_V1").to(device)

    model_transforms = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()

    print("Creating dataloaders...")
    train_dataset = FruitsAndVeggies(os.path.join(path_to_data, "train"), model_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=32)

    validation_dataset = FruitsAndVeggies(os.path.join(path_to_data, "validation"), model_transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, train_dataloader, validation_dataloader, optimizer, device, model_name)



    for epoch in tqdm(range(epochs)):  # Run for 10 epochs
        trainer._run_epoch(epoch)

if __name__ == "__main__":
    main()
