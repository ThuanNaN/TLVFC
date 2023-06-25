import torch 
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def split_dataset(dataset, shuffle_dataset = True, validation_split = 0.2, test_set = False, random_seed= 42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset :
        # np.random.seed(random_seed)
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


