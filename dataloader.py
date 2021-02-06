import torch
from torch.utils.data import Dataset, DataLoader
from dataset import WFLW_Dataset, LoadMode

def create_train_loader(root):
    dataset = WFLW_Dataset(root, mode='train', load_mode=LoadMode.FACE_ONLY)
    dataloader = DataLoader(dataset, 4, shuffle=True)
    return dataloader

def create_test_loader(root):
    dataset = WFLW_Dataset(root, mode='val', load_mode=LoadMode.FULL_IMG)
    dataloader = DataLoader(dataset, 1, shuffle=False)
    return dataloader
