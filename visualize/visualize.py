
import shutil
import os, torch, cv2
import numpy as np

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

def visualize_model(model, dataloader, folder_name):
    if (os.path.exists(folder_name)):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)
    count = 0
    for inputs_in, labels_in, data_package in dataloader:
        inputs = move_to(inputs_in, device)
        labels = move_to(labels_in, device)
        output = model.segment(inputs['image'], inputs['label'])
        for batch_index in range(0, output.shape[0]):
            # # Show image
            # image = inputs['image'][batch_index].clone().detach().cpu().numpy()
            # image = np.moveaxis(image, 0, -1)
            # cv2.imshow('image', image)
            
            # # Show label
            # label = labels['segmentation'][batch_index].clone().detach().cpu().numpy()
            # label = label_to_image(label)
            # cv2.imshow('label', label)

            # # Show output
            # cv2.imshow('output', output[batch_index])
            

            # Write output
            output_instance = crop(output[batch_index], data_package['width'][batch_index], data_package['height'][batch_index])
            cv2.imwrite(folder_name + data_package['image_name'][batch_index] + '.png', output_instance * 255)
            cv2.waitKey(1)
            print("Visualize No: " + str(count), end="\r")
            count += 1
    print()

def visualize(model, dataloader, output_dir):
    model.train()
    model.to(device)

    folder_name = artifact_manager.getDir() + output_dir
    visualize_model(model, dataloader, folder_name)