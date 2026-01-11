"""
Data Loader for IQGD
Wraps MGDM's fluid dataset for RL training
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count


def read_raw_data(path='data/new_dataset'):
    """
    Read raw fluid dataset from MGDM format

    Returns:
        tuple: (cs1_list, cs2_list, history_list, data_list, temps_list)
    """
    names = ['cs1_list', 'cs2_list', 'history_list', 'data_list', 'temps_list']
    full_data_list = [[] for i in range(len(names))]

    for filename in os.listdir(path):
        if filename.endswith('.npy') and 'cs1_list' in filename:
            for dtid, dtname in enumerate(names):
                newfilename = filename.replace("cs1_list", dtname)
                with open(os.path.join(path, newfilename), 'rb') as f:
                    slice_data = torch.tensor(np.load(f), dtype=torch.float32)
                    full_data_list[dtid].append(slice_data)

    return tuple(torch.cat(full_data_list[i], dim=0) for i in range(len(names)))


def normalize_data(cs1_list, cs2_list, init_context_list, data_list):
    """
    Normalize dataset using MGDM's normalization constants

    Returns:
        tuple: Normalized (cs1_list, cs2_list, init_context_list, data_list)
    """
    with torch.no_grad():
        cs1_list_norm = torch.tensor([2.3, 8.4])
        cs2_list_norm = torch.tensor([2.6])
        data_list_norm = torch.tensor([3.2])
        init_context_list_norm = torch.tensor([2.9, 3.6, 2.3, 1.3])

        cs1_list = cs1_list / cs1_list_norm[None, :, None, None]
        cs2_list = cs2_list / cs2_list_norm[None, :, None, None]
        data_list = data_list / data_list_norm[None, :, None, None]
        init_context_list = init_context_list / init_context_list_norm[None, :, None, None]

    return cs1_list, cs2_list, init_context_list, data_list


class IQGDDataset(Dataset):
    """
    Dataset for IQGD training
    Wraps MGDM's SmokeDataset with RL-specific functionality
    """

    def __init__(self, init_context_list, cs1_list, cs2_list, data_list, temps_list):
        super().__init__()
        self.init_context_list = init_context_list
        self.cs1_list = cs1_list
        self.cs2_list = cs2_list
        self.data_list = data_list
        self.temps_list = temps_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        Returns: (initial_context, physics_context_cs1, physics_context_cs2, target_data, timestamps)
        """
        return (
            self.init_context_list[index],
            self.cs1_list[index],
            self.cs2_list[index],
            self.data_list[index],
            self.temps_list[index]
        )


def create_dataloaders(data_path='data/new_dataset', train_ratio=0.9, batch_size=16):
    """
    Create train and test dataloaders for IQGD

    Args:
        data_path: Path to MGDM dataset
        train_ratio: Ratio of training data
        batch_size: Batch size for dataloaders

    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    print("Loading fluid dataset...")
    cs1_list, cs2_list, init_context_list, data_list, temps_list = read_raw_data(data_path)

    print("Normalizing data...")
    cs1_list, cs2_list, init_context_list, data_list = normalize_data(
        cs1_list, cs2_list, init_context_list, data_list
    )

    # Split into train and test
    trainlen = int(len(data_list) * train_ratio)

    train_dataset = IQGDDataset(
        init_context_list[:trainlen],
        cs1_list[:trainlen],
        cs2_list[:trainlen],
        data_list[:trainlen],
        temps_list[:trainlen]
    )

    test_dataset = IQGDDataset(
        init_context_list[trainlen:],
        cs1_list[trainlen:],
        cs2_list[trainlen:],
        data_list[trainlen:],
        temps_list[trainlen:]
    )

    print(f'Dataset loaded: {len(data_list)} samples ({len(train_dataset)} train, {len(test_dataset)} test)')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=min(cpu_count(), 8)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=min(cpu_count(), 8)
    )

    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader, train_ds, test_ds = create_dataloaders()

    print("\nTesting dataloader...")
    for batch in train_loader:
        init_ctx, cs1, cs2, data, temps = batch
        print(f"Initial context shape: {init_ctx.shape}")
        print(f"CS1 shape: {cs1.shape}")
        print(f"CS2 shape: {cs2.shape}")
        print(f"Data shape: {data.shape}")
        print(f"Temps shape: {temps.shape}")
        break

    print("\nData loader test successful!")
