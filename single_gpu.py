from random import shuffle
from typing import List
import torch
from torch.utils.data import DataLoader
import torchvision
from fruits_and_veggies_dataloader_simple import FruitsAndVeggies, FruitsAndVeggiesAugmentator
import os
from tqdm import tqdm
import wandb


with open("/home/ai/.wandb.key", 'r') as wandb_key_file:
    wandb.login(key=wandb_key_file.read().strip(), relogin=True)


path_to_data = "/home/ai/datasets/kaggel/fruits_and_veggies/8/"

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

    def run_epoch(self, epoch):
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

class Tester:
    def __init__(self,
                 model, 
                 test_dataloader,
                 dataset:FruitsAndVeggies,
                 device,
                 experiment_path):
        self.model = model
        self.dataloader = test_dataloader
        self.device = device
        self.experiment_path = experiment_path
        self.dataset = dataset

    def run_test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, label in self.dataloader:
                result = self._run_test_batch(data, label)
                all_preds.append(result['pred'])
                all_labels.append(result['label'])

        self._compute_acc(all_preds, all_labels)




    def _run_test_batch(self, data: torch.Tensor, labels: torch.Tensor):
        data, labels = data.to(self.device), labels.to(self.device)  # Move data and labels to GPU
        pred = self.model(data)
        return {"pred":pred, "label":labels}

    
    def _compute_acc(self, preds:List[torch.Tensor], labels:List):
        label_counts={}
        label_acc_dict={}
        label_pn=torch.zeros( (len(self.dataset.label_name_dict.keys()), len(preds)) )
        
        for key in self.dataset.label_name_dict.keys():
            label_counts[key] = 0

        total_correct = 0
        for i, pred, label in zip(range(len(preds)), preds, labels):
            label_counts[label.item()] += 1
            preds_index = torch.argmax(pred)
            label_pn[label.item(), i] = preds_index == label.item()
            total_correct += preds_index == label.item()

        label_sums = torch.sum(label_pn,dim=1)
        
        label_accs_sum = 0
        for key in self.dataset.label_name_dict.keys():
            n_correct = label_sums[key]
            label_acc = n_correct / label_counts[key]
            label_acc_dict[self.dataset.label_name_dict[key]] = label_acc
            label_accs_sum += label_acc
            print(f"{self.dataset.label_name_dict[key]} acc: {label_acc}" )
        
        macc = label_accs_sum / len(self.dataset.label_name_dict.keys())
        oaacc = total_correct / len(preds)

        wandb.log({"test_macc":macc,
                  "test_oaacc":oaacc,
                  "test_class_acc":label_acc_dict})

        print(f"label positive negative {label_pn}")
        print(f"label sums {label_sums}")
        


def main():
    print("Main entered")
    learning_rate = 0.0001
    learning_rate_scheduler = "non"
    epochs = 5
    batch_size = 16
    model_name = "vit_b_16"
    n_layers_to_freeze = 0
    pretrained_weights = "DEFAULT"
    log_to_wandb = True
    with_augs = False
    augments_to_use = [0]
    if log_to_wandb:
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
                "augementation":augments_to_use,
            },
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available

    print(f"Using device: {device}")

    # Initialize the model and move it to the GPU if available
    model = torchvision.models.vit_b_16(weights=pretrained_weights).to(device)
    num_features = model.heads.head.in_features  # Get the input features of the last layer
    model.heads.head = torch.nn.Linear(num_features, 36).to(device)  # Replace it with a new layer


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
    if with_augs:
        train_augs = FruitsAndVeggiesAugmentator().augementation 
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

    for epoch in tqdm(range(epochs)):  # Run for 10 epochs
        trainer.run_epoch(epoch)

    tester.run_test()


if __name__ == "__main__":
    main()
