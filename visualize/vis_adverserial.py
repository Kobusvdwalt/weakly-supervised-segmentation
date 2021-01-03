
import sys, os, json, torch, cv2
import numpy as np

sys.path.insert(0, os.path.abspath('../'))

from torch.utils.data.dataloader import DataLoader
from data.voc2012_loader_segmentation import PascalVOCSegmentation
from data.voc2012 import label_to_image
from models.unet_adverserial import UNetAdverserial

# Helper function for movement
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device).float()
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")

# Helper function for label
def build_label(transformer, label):
    transformer_vis = transformer[0].clone().detach().cpu().numpy()
    label_vis = np.zeros((transformer_vis.shape[0]+1, transformer_vis.shape[1], transformer_vis.shape[2]))

    # Copy masks based on classification label
    for i in range(0, transformer_vis.shape[0]):
        if label[0, i] > 0.5:
            label_vis[i+1] = transformer_vis[i]


    # Compute background mask
    summed = np.mean(transformer_vis, 0)
    summed[summed > 1] = 1
    summed[summed < 0] = 0
    label_vis[0] = (1 - summed) * 0.5

    label_vis = label_to_image(label_vis)

    return label_vis


model = UNetAdverserial('unet_adverserial')
model.load()
model.to(device)

dataloader = DataLoader(PascalVOCSegmentation('val'), batch_size=4, shuffle=False, num_workers=0)

for inputs_in, labels_in, data_package in dataloader:
    inputs = move_to(inputs_in, device)
    labels = move_to(labels_in, device)
    # Show image
    image = inputs['image'][0].clone().detach().cpu().numpy()
    image = np.moveaxis(image, 0, -1)
    cv2.imshow('image', image)
    
    # Show label
    label = labels['segmentation'][0].clone().detach().cpu().numpy()
    label = label_to_image(label)
    cv2.imshow('label', label)

    # Show output    
    transformer, transformer_clean = model.transformer.segment(inputs['image'], inputs['label'])
    output = build_label(transformer_clean, inputs['label'])
    cv2.imshow('output', output)
    
    cv2.waitKey(0)