from torch.utils.data import DataLoader
from data import PascalVOCDataset
from data import classListArray
from vgg_cam import vgg

import numpy as np
import cv2
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pascal_val = PascalVOCDataset(source='val')
data = DataLoader(pascal_val, batch_size=16, shuffle=True, num_workers=0)

vgg.train()

for images, labels in data:
    