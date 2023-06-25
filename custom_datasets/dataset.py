import os
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from custom_datasets.dataset_utils import split_dataset
from PIL import ImageFile, Image
from scipy.io import loadmat
from skimage import io

import yaml
from yaml.loader import SafeLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True



class Custom_Dataset(Dataset):
    def __init__(self, dir_path,  transform):
        self.dir_path = dir_path
        self.images = datasets.ImageFolder(
                        dir_path, transform=transform
                    )
        self.label2id = self.images.class_to_idx

    
    def __getitem__(self, index):
        image, class_id = self.images[index] 
        return image, class_id

    def __len__(self):
        return len(self.images)



def get_dataloader(data_name, data_root, image_sz, seed, batch_size, num_workers = 2):
    with open('./config/dataset.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    if data_name == "CIFAR10":
        mean = config["dataset_normalize"]["CIFAR10"]["mean"]
        std = config["dataset_normalize"]["CIFAR10"]["std"]
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=tuple(mean),
                std=tuple(std)
            )
        ])
        train_dataset = CIFAR10( data_root , train=True,  transform=data_transforms, download=True)
        test_dataset = CIFAR10( data_root , train=False, transform=data_transforms)
        num_classes = 10

    elif data_name == "PetImages":
        mean = config["dataset_normalize"]["Pet"]["mean"]
        std = config["dataset_normalize"]["Pet"]["std"]
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_sz, image_sz)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=tuple(mean),
                std=tuple(std)
            )
        ])

        pet_dataset = Custom_Dataset(data_root , transform=data_transforms)

        train_sampler, val_sampler, test_sampler = split_dataset(
            pet_dataset, shuffle_dataset=True, random_seed=seed, validation_split=0.2, test_set=0.2)


        dataloaders = {
            "train": DataLoader(pet_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers),
            "val":   DataLoader(pet_dataset, sampler=val_sampler, batch_size=batch_size),
            "test":  DataLoader(pet_dataset, sampler=test_sampler, batch_size=batch_size),
        }
        num_classes = 2
        return num_classes, dataloaders
        
    else:
        mean = config["dataset_normalize"]["Intel"]["mean"]
        std = config["dataset_normalize"]["Intel"]["std"]
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_sz, image_sz)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                    mean=tuple(mean),
                    std=tuple(std)
                )
            ])
        train_dataset = Custom_Dataset(data_root + "/seg_train", transform=data_transforms)
        test_dataset = Custom_Dataset(data_root + "/seg_test", transform=data_transforms)
        num_classes = len(train_dataset.label2id)

    train_sampler, val_sampler, _ = split_dataset(
        train_dataset, shuffle_dataset=True, random_seed=seed, validation_split=0.2, test_set=False)

    dataloaders = {
        "train": DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers),
        "val":   DataLoader(train_dataset, sampler=val_sampler, batch_size=batch_size),
        "test":  DataLoader(test_dataset, batch_size=batch_size),
    }

    return num_classes, dataloaders

