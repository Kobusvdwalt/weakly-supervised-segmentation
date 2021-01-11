import shutil
import sys, os, json, torch, cv2
import numpy as np

sys.path.insert(0, os.path.abspath('../'))

from torch.utils.data.dataloader import DataLoader
from data.voc2012_loader_segmentation import PascalVOCSegmentation
from data.voc2012 import label_to_image
from models.vgg16_gap_feat import Vgg16GAP
from models.unet_adverserial import UNetAdverserial
from training.helpers import move_to

# Helper function for movement
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

    for inputs_in, labels_in, data_package in dataloader:
        inputs = move_to(inputs_in, device)
        labels = move_to(labels_in, device)
        output = model.segment(inputs['image'], inputs['label'])
        for batch_index in range(0, output.shape[0]):
            # Show image
            image = inputs['image'][batch_index].clone().detach().cpu().numpy()
            image = np.moveaxis(image, 0, -1)
            cv2.imshow('image', image)
            
            # Show label
            label = labels['segmentation'][batch_index].clone().detach().cpu().numpy()
            label = label_to_image(label)
            cv2.imshow('label', label)

            # Show output
            cv2.imshow('output', output[batch_index])
            output_instance = crop(output[batch_index], data_package['width'][batch_index], data_package['height'][batch_index])

            # Write output
            cv2.imwrite(folder_name + data_package['image_name'][batch_index] + '.png', output_instance * 255)
            cv2.waitKey(1)


def visualize(model, dataloader):
    model.train()
    model.to(device)

    folder_name = 'output/' + model.name + '/'

    visualize_model(model, dataloader, folder_name)



dataloader = DataLoader(PascalVOCSegmentation('val'), batch_size=4, shuffle=False, num_workers=0)

model = UNetAdverserial(name='unet_adverserial')
model.load()

visualize(model, dataloader)