import sys, os
sys.path.insert(0, os.path.abspath('../'))

from torch.utils.data import DataLoader
from data.loaders import PascalVOCSegmentation
from data.voc2012 import class_list, LabelToImage, LabelToClasses, ClassesToWords, ThresholdClasses, AddBackgroundClass, color_map

from models.vgg_gap import Vgg16

import numpy as np
import cv2
import torch

from PIL import Image

from cam import CAM, GradCAM, GradCAMpp, Gradients

def crop(image, width, height):    
    longest = max(width, height)
    image = cv2.resize(image, (longest, longest))

    starty = int((longest - height) / 2)
    startx = int((longest - width) / 2)


    image = image[starty:longest-starty, startx:longest-startx]
    image = cv2.resize(image, (width, height))
    return image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pascal_val = PascalVOCSegmentation(source='test')
data = DataLoader(pascal_val, batch_size=16, shuffle=False, num_workers=0)

model = Vgg16('multiclass', 20)
model.train()
model.load()
model.to(device)

target_layer = model.conv
# Select visualization
#   wrapped_model = CAM(model, target_layer)
#   wrapped_model = GradCAM(model, target_layer)
#   wrapped_model = GradCAMpp(model, target_layer)
wrapped_model = GradCAM(model, target_layer)

image_count = 0
for images, labels, image_name, image_width, image_height in data:
    inputs = images.permute(0, 3, 1, 2)

    images_np = images.numpy()
    labels_np = labels.numpy()
    image_width_np = image_width.numpy()
    image_height_np = image_height.numpy()

    inputs = inputs.to(device).float()
    labels = labels.to(device).float()

    for sample_index in range(0, len(inputs)):
        sample_input = inputs[sample_index].unsqueeze(0)
        scores, cams = wrapped_model(sample_input)
        # Uncomment this line to use classification labels over predictions
        # scores = LabelToClasses(labels_np[sample_index])[1:]
        # scores = scores[np.newaxis, :]
        # -----------------------------------------------
        final = np.zeros((21, image_height_np[sample_index], image_width_np[sample_index]))
        for class_index in range(0, 20):
            if scores[0, class_index] < 0.5:
                continue
            
            cam_map = cams[class_index]['cam'][0,0].cpu().numpy()
            cam_map = cv2.resize(cam_map, (256, 256), interpolation=cv2.INTER_LINEAR)
            cam_map = crop(cam_map, image_width_np[sample_index], image_height_np[sample_index])
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
        pil_image_p.save('output_multiclass/' + image_name[sample_index] + '.png', 'PNG')

        print('Image No: ' + str(image_count))
        image_count += 1