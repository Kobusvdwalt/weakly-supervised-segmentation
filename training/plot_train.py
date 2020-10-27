import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from datetime import datetime
import json

fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)

legend = []

def plot(file, color, label):
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

    subplot.plot(epochs_val, f1_scores_val, color=color, linestyle='solid', marker='s')
    subplot.plot(epochs_train, f1_scores_train, color=color, linestyle='dashed')

    line = mlines.Line2D([], [], color=color, marker='s', markersize=8, label= label + '_Val')
    line_d = mlines.Line2D([], [], color=color, linestyle='dashed', label=label +'_Train')
    legend.append(line)
    legend.append(line_d)
    



plot('output/log__vgg_11__10-11-2020_13-46-46.txt', 'blue', 'VGG11')
plot('output/log__vgg_13__10-11-2020_13-46-48.txt', 'green', 'VGG13')
plot('output/log__vgg_16__10-11-2020_13-46-51.txt', 'orange', 'VGG16')

# plot('output/log__unfreeze_0__10-11-2020_15-51-26.txt', 'gold', 'Unfrozen_0')
# plot('output/log__unfreeze_1__10-11-2020_15-51-00.txt', 'green', 'Unfrozen_1')
# plot('output/log__unfreeze_2__10-11-2020_15-51-08.txt', 'deeppink', 'Unfrozen_2')
# plot('output/log__unfreeze_3__10-11-2020_15-51-15.txt', 'dodgerblue', 'Unfrozen_3')


subplot.legend(handles=legend)

plt.show()