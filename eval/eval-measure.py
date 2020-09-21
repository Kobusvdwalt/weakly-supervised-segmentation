import sys, os
sys.path.insert(0, os.path.abspath('../'))

from data.loader_factory import LoaderSplit
from models.model_factory import Datasets, Models
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve
from data.voc2012 import class_list

def evaluate_measure(model_enum = Models.Vgg16GAP, dataset_enum = Datasets.voc2012, loader_split = LoaderSplit.val):
    output_name = model_enum.name + '_' + dataset_enum.name + '_' + loader_split.name
    json_file = open('output/raw_' + output_name + '.txt')
    data = json.load(json_file)
    labels = np.asarray(data['labels'])
    outputs = np.asarray(data['outputs'])

    labels = labels.transpose(1, 0)
    outputs = outputs.transpose(1, 0)

    labels[labels > 0.5] = 1
    labels[labels <= 0.5] = 0

    mean_ap = 0
    for class_index in range(0, labels.shape[0]):
        print(class_list[class_index])
        ap = average_precision_score(labels[class_index], outputs[class_index])
        mean_ap += ap
        print(ap)

    mean_ap /= 20
    print('mean')
    print(mean_ap)

    #precision, recall, thresholds = precision_recall_curve(labels[0], outputs[0])
    #plt.figure()
    #plt.step(recall, precision, where='post')
    #plt.show()

evaluate_measure()