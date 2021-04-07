
import matplotlib.pyplot as plt
from artifacts.artifact_manager import artifact_manager
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

def get_grid(images, epochs):
    result = {
        'images': images,
        'epochs': epochs,
        'size_x': len(epochs),
        'size_y': len(images),
        'strips': {}
    }

    for image_index, image in enumerate(images):
        for epoch_index, epoch in enumerate(epochs):
            if epoch == "label":
                im = plt.imread("datasets/voc2012/SegmentationClass/" + image + ".png")
            elif epoch == "image":
                im = plt.imread("datasets/voc2012/JPEGImages/" + image + ".jpg")
            else:
                im = plt.imread(artifact_manager.getDir() + "voc_vgg16_visualization_epoch_" + str(epoch) + "/" + image + ".png")

            result['strips'][str(image_index) + "_" + str(epoch_index)] = im
    return result

def plot_images():
    grid = get_grid(["2007_000033", "2007_000042", "2007_000061", "2007_000123"], [0, 1, 2, "label", "image"])
    
    fig = plt.figure()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)
    fig.suptitle('subplot 1')
    out = fig.subplots(nrows=grid['size_y'], ncols=grid['size_x'])
    
    
    for strip_index, strip in enumerate(out):
        for image_index, image in enumerate(strip):
            image.imshow(grid['strips'][str(strip_index) + "_" + str(image_index)])
            image.set_xticks([])
            image.set_yticks([])
            if image_index == 0:
                image.set_ylabel(grid['images'][strip_index])
            
            if strip_index == grid['size_y']-1:
                if grid['epochs'][image_index] == "label":
                    image.set_xlabel("Ground Truth")
                elif grid['epochs'][image_index] == "image":
                    image.set_xlabel("Image")
                else:
                    image.set_xlabel("epoch " + str(grid['epochs'][image_index]))

    plt.savefig(artifact_manager.getDir() + "weak_plot.png")