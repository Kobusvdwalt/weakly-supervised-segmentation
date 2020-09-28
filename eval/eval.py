
import sys, os, torch, json
import numpy as np
sys.path.insert(0, os.path.abspath('../'))
from torch.utils.data.dataloader import DataLoader
from models.model_factory import Datasets, Models, get_model
from data.loader_factory import LoaderSplit, LoaderType, get_loader



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, dataloader, output_name):
    image_count = 0

    labels_save = []
    outputs_save = []
    names_save = []

    for inputs, labels, names, meta in dataloader:
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()

        outputs = model(inputs)

        outputs_save.append(outputs[0].detach().cpu().numpy())
        labels_save.append(labels[0].detach().cpu().numpy())
        names_save.append(names[0])

        print('Image No: ' + str(image_count))
        image_count += 1

    data = {}
    data['labels'] = labels_save
    data['outputs'] = outputs_save
    data['names'] = names_save

    with open('output/raw_' + output_name + '.txt', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)

def evaluate(model_enum = Models.Vgg16GAP, dataset_enum = Datasets.cityscapes, loader_split = LoaderSplit.val):
    # Set dataset
    dataset = get_loader(dataset_enum, LoaderType.classification, loader_split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Set up model
    model = get_model(dataset_enum, model_enum)
    model.train()
    model.load()
    model.to(device)

    output_name = model_enum.name + '_' + dataset_enum.name + '_' + loader_split.name

    evaluate_model(model, dataloader, output_name)

evaluate()