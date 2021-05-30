from data.voc2012 import color_map, class_list
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from training._common import NumpyEncoder
from datetime import datetime
import json
import numpy as np

def plot_classification_class(epochs, values):
    fig = plt.figure()
    subplot = fig.add_subplot(1, 1, 1)
    legend = []
    values = np.array(values)

    class_colors = color_map(normalized=True)
    class_names = class_list

    for class_index in range(values.shape[1]):
        color_corrected = class_colors[class_index]
        subplot.plot(epochs, values[:, class_index], color=color_corrected, linestyle='solid', marker='s')
        line = mlines.Line2D([], [], color=color_corrected, marker='s', markersize=8, label=class_names[class_index])
        legend.append(line)

    subplot.legend(handles=legend)
    plt.show()

def plot_training(training_log):
    json_file = open(training_log)
    data = json.load(json_file)

    epochs = []
    plots = {}

    for entry in data['train']:
        epochs.append(entry['epoch'])
        for output_key in entry['outputs']:
            output = entry['outputs'][output_key]
            for metric_key in output:
                metric = output[metric_key]
                if output_key + "_" + metric_key not in plots.keys():
                    plots[output_key + "_" + metric_key] = []
                
                plots[output_key + "_" + metric_key].append(metric)

    plot_map = {
        'classification__class_f1': plot_classification_class,
        'segmentation__class_iou': plot_classification_class
    }

    for plot_key in plots:
        if plot_key in plot_map.keys():
            plot_map[plot_key](epochs, plots[plot_key])

def get_list (raw):
    lines = raw.split('\n')
    filtered = []
    for l in lines:
        if l == '':
            continue
        parts = l.split(', ')
        if parts[1] == 'train':
            parts_n = parts[3].split(' ')

            score = parts_n[1]
            loss = parts_n[3]
            filtered.append(float(score))

    return filtered


b_1 =\
"""
epoch    0, train, batch   46, f1 0.0601 loss 0.3305
epoch    1, train, batch   46, f1 0.1480 loss 0.2100
epoch    2, train, batch   46, f1 0.2886 loss 0.1851
epoch    3, train, batch   46, f1 0.3949 loss 0.1692
epoch    4, train, batch   46, f1 0.4597 loss 0.1526
epoch    4, val, batch   46, f1 0.5143 loss 0.1633
epoch    5, train, batch   46, f1 0.5147 loss 0.1432
epoch    6, train, batch   46, f1 0.5529 loss 0.1339
epoch    7, train, batch   46, f1 0.5940 loss 0.1252
epoch    8, train, batch   46, f1 0.6230 loss 0.1175
epoch    8, val, batch   46, f1 0.5147 loss 0.1533
epoch    9, train, batch   46, f1 0.6516 loss 0.1110
epoch   10, train, batch   46, f1 0.6861 loss 0.1015
epoch   11, train, batch   46, f1 0.7053 loss 0.0952
epoch   12, train, batch   46, f1 0.7382 loss 0.0891
epoch   12, val, batch   46, f1 0.5573 loss 0.1611
epoch   13, train, batch   46, f1 0.7407 loss 0.0874
epoch   14, train, batch   46, f1 0.7917 loss 0.0748
epoch   15, train, batch   46, f1 0.7792 loss 0.0746
epoch   16, train, batch   46, f1 0.8171 loss 0.0661
epoch   16, val, batch   46, f1 0.5708 loss 0.1737
epoch   17, train, batch   46, f1 0.8257 loss 0.0638
epoch   18, train, batch   46, f1 0.8267 loss 0.0624
epoch   19, train, batch   46, f1 0.8526 loss 0.0564
epoch   20, train, batch   46, f1 0.8532 loss 0.0545
"""

b_9 =\
"""
epoch    0, train, batch   46, f1 0.0504 loss 0.3252
epoch    1, train, batch   46, f1 0.1421 loss 0.2146
epoch    2, train, batch   46, f1 0.2573 loss 0.1901
epoch    3, train, batch   46, f1 0.3648 loss 0.1717
epoch    4, train, batch   46, f1 0.4199 loss 0.1609
epoch    4, val, batch   46, f1 0.4552 loss 0.1778
epoch    5, train, batch   46, f1 0.4809 loss 0.1502
epoch    6, train, batch   46, f1 0.5238 loss 0.1413
epoch    7, train, batch   46, f1 0.5600 loss 0.1321
epoch    8, train, batch   46, f1 0.5626 loss 0.1295
epoch    8, val, batch   46, f1 0.5025 loss 0.1679
epoch    9, train, batch   46, f1 0.5948 loss 0.1208
epoch   10, train, batch   46, f1 0.6333 loss 0.1137
epoch   11, train, batch   46, f1 0.6618 loss 0.1054
epoch   12, train, batch   46, f1 0.6876 loss 0.1024
epoch   12, val, batch   46, f1 0.5075 loss 0.1714
epoch   13, train, batch   46, f1 0.6875 loss 0.0987
epoch   14, train, batch   46, f1 0.7423 loss 0.0865
epoch   15, train, batch   46, f1 0.7488 loss 0.0842
epoch   16, train, batch   46, f1 0.7544 loss 0.0803
epoch   16, val, batch   46, f1 0.5313 loss 0.1802
epoch   17, train, batch   46, f1 0.7795 loss 0.0749
epoch   18, train, batch   46, f1 0.7946 loss 0.0701
epoch   19, train, batch   46, f1 0.8080 loss 0.0675
epoch   20, train, batch   46, f1 0.8211 loss 0.0645
"""

b_27 =\
"""
epoch    0, train, batch   46, f1 0.0424 loss 0.3158
epoch    1, train, batch   46, f1 0.1096 loss 0.2193
epoch    2, train, batch   46, f1 0.2332 loss 0.1965
epoch    3, train, batch   46, f1 0.3316 loss 0.1805
epoch    4, train, batch   46, f1 0.3542 loss 0.1697
epoch    4, val, batch   46, f1 0.4024 loss 0.1827
epoch    5, train, batch   46, f1 0.4251 loss 0.1604
epoch    6, train, batch   46, f1 0.4700 loss 0.1517
epoch    7, train, batch   46, f1 0.5043 loss 0.1447
epoch    8, train, batch   46, f1 0.5334 loss 0.1363
epoch    8, val, batch   46, f1 0.4457 loss 0.1756
epoch    9, train, batch   46, f1 0.5591 loss 0.1303
epoch   10, train, batch   46, f1 0.5840 loss 0.1252
epoch   11, train, batch   46, f1 0.6119 loss 0.1187
epoch   12, train, batch   46, f1 0.6523 loss 0.1127
epoch   12, val, batch   46, f1 0.4304 loss 0.1801
epoch   13, train, batch   46, f1 0.6695 loss 0.1032
epoch   14, train, batch   46, f1 0.6874 loss 0.1006
epoch   15, train, batch   46, f1 0.7122 loss 0.0959
epoch   16, train, batch   46, f1 0.7232 loss 0.0907
epoch   16, val, batch   46, f1 0.4861 loss 0.1938
epoch   17, train, batch   46, f1 0.7467 loss 0.0856
epoch   18, train, batch   46, f1 0.7532 loss 0.0820
epoch   19, train, batch   46, f1 0.7672 loss 0.0800
epoch   20, train, batch   46, f1 0.7912 loss 0.0709
"""

b_51 =\
"""
epoch    0, train, batch   46, f1 0.0444 loss 0.3205
epoch    1, train, batch   46, f1 0.0412 loss 0.2243
epoch    2, train, batch   46, f1 0.1984 loss 0.2011
epoch    3, train, batch   46, f1 0.2832 loss 0.1870
epoch    4, train, batch   46, f1 0.3501 loss 0.1767
epoch    4, val, batch   46, f1 0.3277 loss 0.1841
epoch    5, train, batch   46, f1 0.3970 loss 0.1672
epoch    6, train, batch   46, f1 0.4338 loss 0.1587
epoch    7, train, batch   46, f1 0.4839 loss 0.1504
epoch    8, train, batch   46, f1 0.5156 loss 0.1423
epoch    8, val, batch   46, f1 0.3972 loss 0.1834
epoch    9, train, batch   46, f1 0.5368 loss 0.1354
epoch   10, train, batch   46, f1 0.5561 loss 0.1300
epoch   11, train, batch   46, f1 0.5818 loss 0.1288
epoch   12, train, batch   46, f1 0.5896 loss 0.1226
epoch   12, val, batch   46, f1 0.3902 loss 0.1957
epoch   13, train, batch   46, f1 0.6270 loss 0.1152
epoch   14, train, batch   46, f1 0.6532 loss 0.1099
epoch   15, train, batch   46, f1 0.6690 loss 0.1058
epoch   16, train, batch   46, f1 0.7033 loss 0.0969
epoch   16, val, batch   46, f1 0.4269 loss 0.2032
epoch   17, train, batch   46, f1 0.7198 loss 0.0928
epoch   18, train, batch   46, f1 0.7358 loss 0.0888
epoch   19, train, batch   46, f1 0.7348 loss 0.0861
epoch   20, train, batch   46, f1 0.7517 loss 0.0838
epoch   20, val, batch   46, f1 0.4465 loss 0.2085
"""

def plot_erase():
    plt.plot(get_list(b_1))
    plt.plot(get_list(b_9))
    plt.plot(get_list(b_27))
    plt.plot(get_list(b_51))

    plt.show()

