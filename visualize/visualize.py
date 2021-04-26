
import shutil
import os, torch, cv2
import numpy as np
import threading

from data.voc2012 import label_to_image
from training.helpers import move_to
from artifacts.artifact_manager import artifact_manager
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helper function for cropping
def crop(image, width, height):    
    longest = max(width, height)
    image = cv2.resize(image, (longest, longest))

    starty = int((longest - height) / 2.0)
    startx = int((longest - width) / 2.0)

    image = image[starty:longest-starty, startx:longest-startx]
    image = cv2.resize(image, (width, height))
    return image

def visualize_model(model, dataloader, folder_name, max_count):
    if (os.path.exists(folder_name)):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)
    threads = []
    count = 0
    for inputs_in, labels_in, data_package in dataloader:
        inputs = move_to(inputs_in, device)
        labels = move_to(labels_in, device)
        outputs = model.segment(inputs['image'], inputs['label'])
        for batch_index in range(0, outputs.shape[0]):
            output_instance = crop(outputs[batch_index], data_package['width'][batch_index], data_package['height'][batch_index])
            cv2.imwrite(folder_name + data_package['image_name'][batch_index] + '.png', output_instance * 255)

            print('visualize', count, end='\r')
            count += 1
        if count > max_count:
            break
    print()

def visualize(model, dataloader, output_dir, max_count=1_000_000):
    model.train()
    model.to(device)

    folder_name = artifact_manager.getDir() + output_dir
    visualize_model(model, dataloader, folder_name, max_count)