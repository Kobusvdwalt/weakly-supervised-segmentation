import sys, os, torch
from torch.utils.data.dataloader import DataLoader
from training.train import train
from models.gain_unet import Gain_UNET

from data.voc2012_loader_classification import PascalVOCClassification
def start():
    # VGG16
    model = Gain_UNET(name='voc_gain_unet')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=32, shuffle=True, num_workers=8),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=32, shuffle=False, num_workers=8)
        },
        epochs=101,
    )