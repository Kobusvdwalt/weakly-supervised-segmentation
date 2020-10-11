import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from datetime import datetime
import json

# json_file = open('output/log__unfreeze_0__10-11-2020_12-37-05.txt')
# json_file = open('output/log__unfreeze_1__10-11-2020_12-37-35.txt')
# json_file = open('output/log__unfreeze_2__10-11-2020_12-37-36.txt')

def plot(file):
    json_file = open(file)

    data = json.load(json_file)

    # Training time
    date_time_obj_start = datetime.strptime(data['training_start'], '%d-%m-%Y_%H-%M-%S')
    date_time_obj_end = datetime.strptime(data['training_update'], '%d-%m-%Y_%H-%M-%S')
    print('Total training time :')
    print(date_time_obj_end - date_time_obj_start)

    epochs_train = []
    f1_scores_train = []

    for entry in data['train']:
        f1_scores_train.append(entry['f1'])
        epochs_train.append(entry['epoch'])

    epochs_val = []
    f1_scores_val = []
    for entry in data['val']:
        f1_scores_val.append(entry['f1'])
        epochs_val.append(entry['epoch'])

    fig = plt.figure()
    subplot = fig.add_subplot(1, 1, 1)

    subplot.plot(epochs_val, f1_scores_val, color='blue', linestyle='solid', marker='s')
    subplot.plot(epochs_train, f1_scores_train, color='blue', linestyle='dashed')

    #plt.plot(epochs, f1_scores_3, color='green', linestyle='solid', marker='s')
    #plt.plot(epochs, f1_scores_4, color='green', linestyle='dashed')

    blue_line = mlines.Line2D([], [], color='blue', marker='s', markersize=8, label='Trainable 1 - Val')
    blue_line_d = mlines.Line2D([], [], color='blue', linestyle='dashed', label='Trainable 1 - Train')

    geen_line = mlines.Line2D([], [], color='green', marker='s', markersize=8, label='Trainable 2 - Val')
    geen_line_d = mlines.Line2D([], [], color='green', linestyle='dashed', label='Trainable 2 - Train')
    subplot.legend(handles=[blue_line, blue_line_d, geen_line, geen_line_d])


    

plot('output/log__unfreeze_0__10-11-2020_12-37-05.txt')
plot('output/log__unfreeze_1__10-11-2020_12-37-35.txt')
plot('output/log__unfreeze_2__10-11-2020_12-37-36.txt')
plot('output/log__unfreeze_3__10-11-2020_12-38-03.txt')

plt.show()