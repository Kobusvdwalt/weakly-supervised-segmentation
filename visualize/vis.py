
import sys, os, torch
import numpy as np
import cv2
import shutil

from torch.utils.data.dataloader import DataLoader
from PIL import Image

sys.path.insert(0, os.path.abspath('../'))


import data.voc2012 as voc2012
from data.voc2012 import color_map
from data.loader_factory import LoaderSplit, LoaderType, get_loader
from models.model_factory import Datasets, Models, get_model
from visualize.cam import CAM, GradCAM, GradCAMpp, Gradients, SmoothGradCAMpp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def crop(image, width, height):    
    longest = max(width, height)
    image = cv2.resize(image, (longest, longest))

    starty = int((longest - height) / 2)
    startx = int((longest - width) / 2)


    image = image[starty:longest-starty, startx:longest-startx]
    image = cv2.resize(image, (width, height))
    return image


# Set dataset
dataset_enum = Datasets.voc2012
dataset = get_loader(dataset_enum, LoaderType.segmentation, LoaderSplit.val)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Set up model
model_enum = Models.Vgg16GAP
model = get_model(dataset_enum, model_enum)
model.train()
model.load()
model.to(device)

# Set target layer
target_layer = model.conv

# Set up wrapped model
wrapped_model = GradCAM(model, target_layer)

folder_name = 'output/' + model_enum.name + '_' + dataset_enum.name + '_' + LoaderSplit.val.name
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
os.mkdir(folder_name)

image_count = 0
for images, labels, image_name, image_width, image_height in dataloader:
    # Show image and label
    # image_np = images[0].numpy()
    # label_np = voc2012.label_to_image(labels[0].numpy())

    # cv2.imshow('image', image_np)
    # cv2.imshow('label', label_np)
    # cv2.waitKey(0)

    # Permute for prediction
    inputs = images.permute(0, 3, 1, 2)
    inputs = inputs.to(device).float()
    labels = labels.to(device).float()

    # Predict
    scores, cams = wrapped_model(inputs)
    scores_np = scores[0].detach().cpu().numpy()

    class_count_np = scores_np.shape[0]
    image_name_np = image_name[0]
    image_width_np = image_width[0].numpy()
    image_height_np = image_height[0].numpy()

    final = np.zeros((class_count_np+1, image_height_np, image_width_np))
    for class_index in range(0, class_count_np):
        if scores_np[class_index] < 0.5:
            continue
        
        cam_map = cams[class_index]['cam'][0,0].cpu().numpy()
        cam_map = cv2.resize(cam_map, (256, 256), interpolation=cv2.INTER_LINEAR)
        cam_map = crop(cam_map, image_width_np, image_height_np)
        cam_map = np.array(cam_map * 255, dtype = np.uint8)
        ret, cam_map = cv2.threshold(cam_map, 50, 255, cv2.THRESH_BINARY)
        final[class_index+1] = cam_map

    summed = np.sum(final, 0)
    summed[summed > 1] = 1
    summed[summed < 0] = 0
    
    final[0] = 1 - summed

    indexmap = np.argmax(final, axis=0)
    indexmap = indexmap.astype(dtype=np.uint8)
    pil_image_p = Image.fromarray(indexmap)
    palette = color_map(256)
    pil_image_p.putpalette(palette)

    

    pil_image_p.save(folder_name  + '/' + image_name_np + '.png', 'PNG')

    print('Image No: ' + str(image_count))
    image_count += 1