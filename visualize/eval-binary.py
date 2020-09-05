import sys, os
sys.path.insert(0, os.path.abspath('../'))

from torch.utils.data import DataLoader
from data.loaders import PascalVOCSegmentation
from data.voc2012 import class_list, class_list_classification, LabelToImage, ImageToLabel, LabelToClasses, ClassesToWords, ThresholdClasses, AddBackgroundClass

from models.vgg_gap import Vgg16

import numpy as np
import cv2
import torch

from cam import CAM, GradCAM, GradCAMpp, Gradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pascal_val = PascalVOCSegmentation(source='val')
data = DataLoader(pascal_val, batch_size=16, shuffle=False, num_workers=0)


image_count = 0
for images, labels, imageNames in data:
    inputs = images.permute(0, 3, 1, 2)

    images_np = images.numpy()
    labels_np = labels.numpy()

    inputs = inputs.to(device).float()
    labels = labels.to(device).float()

    for sample_index in range(0, len(inputs)):
        sample_input = inputs[sample_index].unsqueeze(0)
        final = np.zeros((21, 256, 256))
        for class_name in class_list_classification:
            model = Vgg16('binary_' + class_name, 2)
            model.train()
            model.load()
            model.to(device)

            target_layer = model.conv
            # Select visualization
            #   wrapped_model = CAM(model, target_layer)
            #   wrapped_model = GradCAM(model, target_layer)
            #   wrapped_model = GradCAMpp(model, target_layer)
            wrapped_model = GradCAM(model, target_layer)
            scores, cams = wrapped_model(sample_input)
        
            if scores[0, 0] < 0.5:
                continue

            cam_map = cams[0]['cam'][0,0].cpu().numpy()
            cam_map = cv2.resize(cam_map, (256, 256), interpolation=cv2.INTER_LINEAR)
            cam_map = np.array(cam_map * 255, dtype = np.uint8)
            ret, cam_map = cv2.threshold(cam_map, 50, 255, cv2.THRESH_BINARY)
            
            class_index = class_list.index(class_name)
            final[class_index] = cam_map

        summed = np.sum(final, 0)
        summed[summed > 1] = 1
        summed[summed < 0] = 0
        
        final[0] = 1 - summed

        image = LabelToImage(final)
        cv2.imwrite('output_binary/' + imageNames[sample_index] + '.png', image * 255)

        print('Image No: ' + str(image_count))
        image_count += 1