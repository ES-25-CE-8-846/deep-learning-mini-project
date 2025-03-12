
import torch 
import torchvision 
from torchinfo import torchinfo
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os


path_to_data = "/home/a/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8"


class FruitsAndVeggies(Dataset):
    def __init__(self, split_root):
        # create a dict of labels and filepaths
        class_dir_names = sorted(os.listdir(split_root))
        n_classes = len(class_dir_names)
        # create one-hot encoding 
        self.dataset_list = []
        for i, class_dir in enumerate(class_dir_names):
            label = torch.zeros(n_classes)
            label[i] = 1
            for image in sorted(os.listdir(os.path.join(split_root, class_dir))):
                self.dataset_list.append([label, os.path.join(split_root,class_dir,image)])
        
    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        data_list = self.dataset_list[index]
        image_path = data_list[1]
        label = data_list[0]
        image = torchvision.io.read_file(image_path)
        
        return image, label
    



test = FruitsAndVeggies(os.path.join(path_to_data,"train"))

for data in test:
    print(data)





