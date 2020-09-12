
# TODO: NOW WE CAN MEASURE ALL OF THE DETAILS OF THE CITYSCAPES DATASET
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import cv2
sys.path.insert(0, os.path.abspath('../'))

from data.cityscapes import get_class_count, get_label_words, get_labels, words_to_classes


split = 'train'

file = open('./output/cityscapes_classification_' + split + '.txt', 'r')
lines = file.readlines()

class_count_store = np.zeros(shape=(get_class_count()))
for line in lines:
    image_path = line.split(' ')[0].replace('\n', '')
    label_words = line.split(' ')[1].replace('\n', '')
    classes = words_to_classes(label_words)
    class_count_store += classes

    # Which images don't have "ROAD"
    # if (classes[0] == 0):
    #     label_path = image_path.replace('_leftImg8bit', '_gtFine_color')
    #     label_path = label_path.replace('leftImg8bit', 'gtFine')

    #     image = cv2.imread('../datasets/cityscapes/' + image_path)
    #     label = cv2.imread('../datasets/cityscapes/' + label_path)

image_count = len(lines)
print('Image Count:')
print(image_count)

for index, label in enumerate(get_labels()):
    label_count = class_count_store[index]
    print('------------------------')
    print('Label: ' + label.name)
    print('Count: ' + str(label_count))
    print('Frequency: ' + str(label_count / image_count))

bar_data = get_label_words()
bar_data.insert(0, 'Total')
class_count_store = np.insert(class_count_store, 0, image_count)
x_pos = [i for i, _ in enumerate(bar_data)]
plt.bar(x_pos, class_count_store)
plt.xticks(x_pos, bar_data, rotation='vertical')

plt.show()