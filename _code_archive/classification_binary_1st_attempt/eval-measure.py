import sys, os
sys.path.insert(0, os.path.abspath('../'))

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve
from data.voc2012 import class_list_classification

json_file = open('output/data_val.txt')
data = json.load(json_file)

mean_ap = 0
for class_name in class_list_classification:
    print(class_name)
    labels = np.asarray(data['labels_'+ class_name])
    outputs = np.asarray(data['outputs_'+ class_name])

    labels = labels.transpose(1, 0)
    outputs = outputs.transpose(1, 0)

    labels[labels > 0.5] = 1
    labels[labels <= 0.5] = 0

    ap = average_precision_score(labels[0], outputs[0])
    mean_ap += ap
    print(ap)

mean_ap /= 20
print('mean')
print(mean_ap)


#precision, recall, thresholds = precision_recall_curve(labels[0], outputs[0])
#plt.figure()
#plt.step(recall, precision, where='post')
#plt.show()
