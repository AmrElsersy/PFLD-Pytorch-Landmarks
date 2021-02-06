import torch
from torch.utils.data import Dataset, DataLoader
from dataset import WFLW_Dataset, LoadMode

def create_train_loader(root='data/WFLW', batch_size = 64, mode = LoadMode.FACE_ONLY):
    dataset = WFLW_Dataset(root, mode='train', load_mode=mode)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader

def create_test_loader(root='data/WFLW', batch_size = 1, mode = LoadMode.FULL_IMG):
    dataset = WFLW_Dataset(root, mode='val', load_mode=mode)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    return dataloader
