from data.voc2012 import color_map, class_list
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from training.helpers import NumpyEncoder
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

    

    
# def plot_training 
    
