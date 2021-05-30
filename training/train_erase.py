import sys, os, torch
from torch.utils.data.dataloader import DataLoader
from training.train import train
from models.wass import WASS

from data.voc2012_loader_classification import PascalVOCClassification
def start():
    # VGG16
    model = WASS('voc_wass')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=32, shuffle=True, num_workers=8),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=32, shuffle=False, num_workers=8)
        },
        epochs=101,
    )