import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import json

json_file = open('output/log_10-11-2020_09-18-18.txt')
data = json.load(json_file)

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

plt.plot(epochs_val, f1_scores_val, color='blue', linestyle='solid', marker='s')
plt.plot(epochs_train, f1_scores_train, color='blue', linestyle='dashed')

#plt.plot(epochs, f1_scores_3, color='green', linestyle='solid', marker='s')
#plt.plot(epochs, f1_scores_4, color='green', linestyle='dashed')

blue_line = mlines.Line2D([], [], color='blue', marker='s', markersize=8, label='Trainable 1 - Val')
blue_line_d = mlines.Line2D([], [], color='blue', linestyle='dashed', label='Trainable 1 - Train')

geen_line = mlines.Line2D([], [], color='green', marker='s', markersize=8, label='Trainable 2 - Val')
geen_line_d = mlines.Line2D([], [], color='green', linestyle='dashed', label='Trainable 2 - Train')
plt.legend(handles=[blue_line, blue_line_d, geen_line, geen_line_d])


plt.show()
