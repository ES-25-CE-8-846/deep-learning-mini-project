
import torch 
import torchvision 
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2 
from PIL import Image 



class FruitsAndVeggies(Dataset):
    def __init__(self, split_root, transforms):
        # create a dict of labels and filepaths
        class_dir_names = sorted(os.listdir(split_root))
        n_classes = len(class_dir_names)
        
        print(n_classes)
        
        self.label_name_dict = {}

        for label, class_name in enumerate(class_dir_names):
            self.label_name_dict[label] = class_name

        self.transforms = transforms
        # create one-hot encoding 
        self.dataset_list = []
        for i, class_dir in enumerate(class_dir_names):
            label = i
            
            extension_set = {"jpg", "png",  "JPG", "jpeg"}

            for image in sorted(os.listdir(os.path.join(split_root, class_dir))):
                extension = image.split(".")[-1]

                if extension in extension_set:
                    self.dataset_list.append([label, os.path.join(split_root,class_dir,image)])
                else:
                    print(f"{extension} found in dataset")
        
    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        data_list = self.dataset_list[index]
        image_path = data_list[1]
        label = data_list[0]
        image = Image.open(image_path)       

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transforms(image)

        return image, label
    

if __name__ =="__main__":

    path_to_data = "/home/ai/datasets/kaggel/fruits_and_veggies/8/"
    test = FruitsAndVeggies(os.path.join(path_to_data,"validation"), torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms())
    
    data_loader = DataLoader(test, batch_size=32, shuffle=True)

    # for data, label in data_loader:
    #     print(label)
    #
    # for data in test:
    #     image, label = data

    print(test.label_name_dict)
        # print(image.size)

