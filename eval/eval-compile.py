import sys, os
sys.path.insert(0, os.path.abspath('../'))

import json
import numpy as np
from data.voc2012 import class_list

json_file = open('data_val.txt')

data = json.load(json_file)
outputs = np.asarray(data['outputs'])
imageNames = data['imageNames']

outputs = outputs.transpose(1, 0)

for class_index in range(0, outputs.shape[0]):
    textFile = open('./test/comp1_cls_test_' + class_list[class_index] + '.txt', 'w')
    for sample_index in range(0, outputs.shape[1]):
        textFile.write(imageNames[sample_index] + ' ' + '{:.6f}'.format(outputs[class_index, sample_index]) +'\n')
    textFile.close()