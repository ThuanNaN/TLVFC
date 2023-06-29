import os
import glob
from pathlib import Path
import torch 
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def split_dataset(dataset, shuffle_dataset = True, validation_split = 0.2, test_set = False, random_seed= 42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset :
        np.random.shuffle(indices)

    if test_set:
        val_split = int(np.floor(validation_split * dataset_size)) 
        test_split = int(np.floor(test_set * dataset_size)) 
        val_indices, test_indices, train_indices =  indices[:val_split], \
                                                    indices[val_split:val_split + test_split], \
                                                    indices[val_split + test_split:]
    else:
        split = int(np.floor(validation_split * dataset_size)) 
        val_indices, test_indices, train_indices =  indices[:split], \
                                                    None, \
                                                    indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    if test_indices is not None:
        test_sampler = SubsetRandomSampler(test_indices)
    else:
        test_sampler = None
    
    return train_sampler, valid_sampler, test_sampler


def split_data_dir(dir_root: str, save_foler: str, ratio_split = 0.2, random_seed = 42):
    np.random.seed(random_seed)
    src_dir = Path(dir_root)
    classes_list = os.listdir(src_dir)

    save_dir = Path(save_foler)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    new_train_set = save_dir / "new_train"
    new_train_set.mkdir(exist_ok=True, parents=True)

    new_val_set = save_dir / "new_val"
    new_val_set.mkdir(exist_ok=True, parents=True)
    
    for cls in classes_list:
        data_dir = src_dir / cls
        data_list = glob.glob(str(data_dir / "*.jpg"))

        num_data = len(data_list)
        indices = list(range(num_data))
        split = int(np.floor(ratio_split * num_data))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = [data_list[i] for i in train_idx]
        valid_sampler = [data_list[i] for i in valid_idx]

        new_train_cls = new_train_set / cls
        new_train_cls.mkdir(exist_ok=False, parents=True)
        
        new_val_cls = new_val_set / cls
        new_val_cls.mkdir(exist_ok=False, parents=True)
        

        for i in train_sampler:
            cmd = f"cp {i} {new_train_cls}"
            os.system(cmd)
        

        for i in valid_sampler:
            cmd = f"cp {i} {new_val_cls}"
            os.system(cmd)



def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        try:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1
        except Exception as e:
            print(e)
        
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return {
        "mean": mean,
        "std": std
    }


if __name__ == "__main__":

    # split_data_dir("./data/Intel/seg_train", save_foler="./data/Intel",ratio_split = 0.2, random_seed=2)

    # split_data_dir("./data/PetImages", save_foler="./data/PetImages", ratio_split = 0.2, random_seed=2)
    # split_data_dir("./data/PetImages/seg_train", save_foler="./data/PetImages", ratio_split = 0.2,  random_seed=2)



    pass
