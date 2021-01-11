from data.loader_factory import LoaderSplit
from models.model_factory import Datasets, Models
import sys, os
sys.path.insert(0, os.path.abspath('../'))

import json
import numpy as np
from data.voc2012 import class_list


def evaluate_compile(model_enum = Models.Vgg16GAP, dataset_enum = Datasets.voc2012, loader_split = LoaderSplit.val):
    output_name = model_enum.name + '_' + dataset_enum.name + '_' + loader_split.name
    json_file = open(output_name + '.txt')
    data = json.load(json_file)
    imageNames = data['names']
    outputs = np.asarray(data['outputs'])
    outputs = outputs.transpose(1, 0)

    for class_index in range(0, outputs.shape[0]):
        textFile = open('./test/comp1_cls_test_' + class_list[class_index] + '.txt', 'w')
        for sample_index in range(0, outputs.shape[1]):
            textFile.write(imageNames[sample_index] + ' ' + '{:.6f}'.format(outputs[class_index, sample_index]) +'\n')
        textFile.close()

evaluate_compile