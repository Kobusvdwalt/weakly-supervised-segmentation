#
# This generates a set of text files to organize the data as needed
#
import sys, os, cv2
sys.path.insert(0, os.path.abspath('../'))

from data.cityscapes import classes_to_words, image_to_classes, image_to_label

# **********************************************************
# Segmentation
def segmentation_generation(split):
    text_file = open('./output/cityscapes_segmentation_' + split + '.txt', 'w')
    cities = os.listdir('../datasets/cityscapes/leftImg8bit/' + split + '/')
    for city in cities:
        files = os.listdir('../datasets/cityscapes/leftImg8bit/' + split + '/' + city + '/')
        for file in files:
            text_file.write('leftImg8bit/' + split + '/' + city + '/' + file + ' ')
            text_file.write('gtFine/' + split + '/' + city + '/' + file.replace('leftImg8bit', 'gtFine_color') +'\n')
    text_file.close()

segmentation_generation('train')
segmentation_generation('val')
segmentation_generation('test')

# **********************************************************
# Classification
def classification_generation(split):
    count = 0
    segmentation_file = open('./output/cityscapes_segmentation_' + split + '.txt', 'r')
    classification_file = open('./output/cityscapes_classification_' + split + '.txt', 'w')
    lines = segmentation_file.readlines()
    for line in lines:
        # Get the input and label paths
        image_file = line.split(' ')[0].replace('\n', '')
        label_file = '../datasets/cityscapes/' +  line.split(' ')[1].replace('\n', '')
        print(str(count) + '/' + str(len(lines)))
        count += 1
        # We read the label color image
        label = cv2.imread(label_file)
        # Resize for performance reasons
        label = cv2.resize(label, (256, 128))
        classes = image_to_classes(label)
        words = classes_to_words(classes)[1:]

        # Write the input path and classification label
        classification_file.write(image_file + ' ')
        classification_file.write('|'.join(words) + '\n')

    classification_file.close()
classification_generation('train')
classification_generation('val')
classification_generation('test')



