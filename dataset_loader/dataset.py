from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from dataset_loader.dataset_utils import split_dataset

import yaml
from yaml.loader import SafeLoader

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


def get_train_valid_loader(dataset_name,
                            data_dir, 
                           batch_size, 
                           augment, 
                           random_seed, 
                           valid_size = 0.2, 
                           shuffle=True, 
                           num_workers = 1, 
                           pin_memory=False):
    
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    with open('./config/dataset.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    mean = config["dataset_config"][dataset_name]["mean"]
    std = config["dataset_config"][dataset_name]["std"]
    image_size = config["dataset_config"][dataset_name]["image_size"]

    normalize = transforms.Normalize(
            mean=tuple(mean),
            std=tuple(std)
    )
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

        train_sampler, val_sampler, _ = split_dataset(
            train_dataset, shuffle_dataset=shuffle, random_seed=random_seed, validation_split=0.2, test_set=False)

        train_loader = DataLoader(train_dataset, 
                                  sampler=train_sampler, 
                                  batch_size=batch_size, 
                                  num_workers=num_workers,
                                  pin_memory = pin_memory)
        valid_loader = DataLoader(valid_dataset,
                                  sampler=val_sampler, 
                                  batch_size=batch_size, 
                                  pin_memory=pin_memory)

        return (train_loader, valid_loader)
    
    else:
        train_dataset = Custom_Dataset(data_dir +"/new_train" , 
                                       transform=train_transform)
        valid_dataset = Custom_Dataset(data_dir + "/new_val" , 
                                       transform=valid_transform)

        train_loader = DataLoader(train_dataset, 
                                  shuffle=True,
                                  batch_size=batch_size, 
                                  num_workers=num_workers,
                                  pin_memory = pin_memory)
        
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  pin_memory=pin_memory)

        return (train_loader, valid_loader)


def get_test_loader(
                    dataset_name,
                    data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False):

    with open('./config/dataset.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    mean = config["dataset_config"][dataset_name]["mean"]
    std = config["dataset_config"][dataset_name]["std"]
    image_size = config["dataset_config"][dataset_name]["image_size"]

    normalize = transforms.Normalize(
            mean=tuple(mean),
            std=tuple(std)
    )

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
    else:
        dataset = Custom_Dataset(data_dir + "/seg_test", 
                                 transform=transform)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    return data_loader
