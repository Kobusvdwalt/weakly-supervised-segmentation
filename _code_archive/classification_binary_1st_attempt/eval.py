if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    from torch.utils.data import DataLoader
    from data.loaders import PascalVOCClassificationBinary
    from data.voc2012 import class_list_classification
    from models.vgg_gap import Vgg16

    import torch
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import numpy as np
    import torchvision
    from torchvision import datasets, models, transforms
    import matplotlib.pyplot as plt
    import time
    import json
    import copy

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data = {}

    for class_name in class_list_classification:
        target_class = class_name
        print(target_class)

        # Set up dataloader
        source = 'val'
        pascal_val = PascalVOCClassificationBinary(source=source, target=target_class)
        dataloader = DataLoader(pascal_val, batch_size=16, shuffle=False, num_workers=6)

        def eval_model(model, criterion):
            model.eval()

            tp = torch.zeros(2).to(device)
            tn = torch.zeros(2).to(device)
            fp = torch.zeros(2).to(device)
            fn = torch.zeros(2).to(device)

            labels_save = []
            outputs_save = []
            imageNames_save = []

            batch_count = 0
            for inputs, labels, imageNames in dataloader:
                batch_count += 1
                inputs = inputs.permute(0, 3, 1, 2)
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                for i in range(0, inputs.shape[0]):
                    outputs_save.append(outputs[i].detach().cpu().numpy())
                    labels_save.append(labels[i].detach().cpu().numpy())
                    imageNames_save.append(imageNames[i])

                print(' Batch: {} '.format(batch_count), end='\r')

            
            data['labels_' + target_class] = labels_save
            data['outputs_' + target_class] = outputs_save
            data['imageNames_' + target_class] = imageNames_save

        vgg = Vgg16('binary_' + target_class, 2)
        vgg.load()
        vgg.to(device)
        eval_model(vgg, torch.nn.BCELoss())

    with open('output/data_' +source+ '.txt', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)