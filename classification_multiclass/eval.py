if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    from torch.utils.data import DataLoader
    from data.loaders import PascalVOCClassificationMulticlass
    from models.vgg_gap_up_p import Vgg16

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
    
    # Set up dataloader
    source = 'val'
    pascal_val = PascalVOCClassificationMulticlass(source=source)
    dataloader = DataLoader(pascal_val, batch_size=16, shuffle=False, num_workers=6)

    def eval_model(model, criterion):
        model.eval()

        tp = torch.zeros(20).to(device)
        tn = torch.zeros(20).to(device)
        fp = torch.zeros(20).to(device)
        fn = torch.zeros(20).to(device)

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

        data = {}
        data['labels'] = labels_save
        data['outputs'] = outputs_save
        data['imageNames'] = imageNames_save

        with open('output/data_' + source + '.txt', 'w') as outfile:
            json.dump(data, outfile, cls=NumpyEncoder)
    
    vgg = Vgg16('multiclass_up', 20)
    vgg.load()
    vgg.to(device)

    eval_model(vgg, torch.nn.BCELoss())