
import torch 
from torch._prims_common import Tensor
import torchvision 
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image 
import torchvision
from torchvision.transforms import transforms


class FruitsAndVeggies(Dataset):
    def __init__(self, split_root, transforms, augmentations = None):
        # create a dict of labels and filepaths
        class_dir_names = sorted(os.listdir(split_root))

        n_classes = len(class_dir_names)
        
        print(n_classes)
        
        self.label_name_dict = {}

        self.augmentations = augmentations

        for label, class_name in enumerate(class_dir_names):
            self.label_name_dict[label] = class_name


        self.n_classes = len(class_dir_names)

        self.transforms = transforms
        
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

        if self.augmentations != None:
             image = self.augmentations(image)

        return image, label

class FruitsAndVeggiesAugmentator:
    def __init__(self, transforms_to_use = 'all'):
        transform_list = [
            torchvision.transforms.ColorJitter(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomErasing(),
            torchvision.transforms.RandomRotation(degrees=90),
            None
        ]
        
        transform_to_index_dict={
                'all':[0,1,2,3],
                'ColorJitter':[0],
                'RandomHorizontalFlip':[1],
                'RandomErasing':[2],
                'RandomRotation':[3],
                'none':[],
                }
        
        print(transform_to_index_dict.keys())

        transforms_selected = []

        for transform_index in transform_to_index_dict[transforms_to_use]:
            
            transforms_selected.append(transform_list[transform_index])

        if len(transforms_selected) > 0:
            self.augementation = torchvision.transforms.Compose(transform_list)
        else:
            self.augementation = None




if __name__ =="__main__":
    import matplotlib.pyplot as plt 
    import numpy as np

    path_to_data = "/home/ai/datasets/kaggel/fruits_and_veggies/8/"

    augmentations = FruitsAndVeggiesAugmentator().augementation

    test_aug = FruitsAndVeggies(os.path.join(path_to_data,"validation"), torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms(), augmentations)
    
    data_loader = DataLoader(test_aug, batch_size=1, shuffle=True)
    
    for i in range(20):

        img, label = test_aug[2]
        img = np.array(img)
        plt.imshow(img.transpose(1,2,0))
        plt.show()

    # for data, label in data_loader:
    #     print(label)
    #
    # for data in test:
    #     image, label = data

        # print(image.size)



