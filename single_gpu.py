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
    def __init__(self,
                 model,
                 train_dataloader,
                 validation_dataloader,
                 optimizer: torch.optim.Optimizer,
                 device,
                 model_name,
                 experiment_path):
        self.model = model.to(device)  # Move model to the device (GPU)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.current_epoch = 0
        self.model_name = model_name
        self.experiment_path = experiment_path

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
        return {"pred":pred, "label":labels, "loss":loss.item()}

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

        # Validate epoch
        self.model.eval()
        validation_loss = 0
        validation_acc = 0
        total_size = 0
        for data, labels in self.validation_dataloader:
            batch_output_dict = self._run_validation_batch(data, labels)
            validation_acc += self._compute_acc(batch_output_dict["pred"], batch_output_dict["label"]) * batch_output_dict["label"].shape[0]
            total_size += batch_output_dict["label"].shape[0]
            validation_loss += batch_output_dict["loss"]
        validation_acc = validation_acc / total_size
        validation_loss = validation_loss / len(self.validation_dataloader)

        wandb.log({"val_loss":validation_loss, "val_acc":validation_acc * 100})

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
        path = os.path.join("checkpoints", self.experiment_path)
        os.makedirs(path, exist_ok=True)
        file_name = f"{self.current_epoch}_".rjust(5,"0") + f"{self.model_name}_{round(self.last_val_loss, 3)}__{round(self.best_val_loss, 3)}.pth"

        full_save_path = os.path.join(path, file_name)
        torch.save(self.model.state_dict(), full_save_path)



def main():
    print("Main entered")
    learning_rate = 0.0001
    learning_rate_scheduler = "non"
    epochs = 10
    batch_size = 16
    model_name = "vit_b_16"
    n_layers_to_freeze = 0
    pretrained_weights = "DEFAULT"
    log_to_wandb = True

    if log_to_wandb:
        run = wandb.init(name=f"a_bs{batch_size}_{model_name}_freeze_{n_layers_to_freeze}",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available

    print(f"Using device: {device}")

    # Initialize the model and move it to the GPU if available
    model = torchvision.models.vit_b_16(weights=pretrained_weights).to(device)

    # Freeze model wights
    parameter_index = 0
    for layer in model.children():
        for param in layer.parameters():
            if parameter_index < n_layers_to_freeze:
                print(f"freezing parameter with index {parameter_index}")
                param.requires_grad = False
            else:
                print(f"did not freeze {parameter_index}")
            parameter_index += 1

    model_transforms = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()

    print("Creating dataloaders...")
    train_dataset = FruitsAndVeggies(os.path.join(path_to_data, "train"), model_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=32)

    validation_dataset = FruitsAndVeggies(os.path.join(path_to_data, "validation"), model_transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(model = model,
                      train_dataloader = train_dataloader,
                      validation_dataloader = validation_dataloader,
                      optimizer = optimizer,
                      device = device,
                      model_name = model_name,
                      experiment_path = f"a_bs{batch_size}_{model_name}_freeze_{n_layers_to_freeze}")


    for epoch in tqdm(range(epochs)):  # Run for 10 epochs
        trainer._run_epoch(epoch)

if __name__ == "__main__":
    main()
