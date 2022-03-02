import torch, json
import numpy as np
from datetime import datetime
from artifacts.artifact_manager import artifact_manager

def move_to(obj, device):
    if torch.is_tensor(obj):
        if obj.dtype == torch.int64:
            return obj.to(device)
        else:
            return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")

def get_metric(file_path, metric_key, phase='train'):
    json_file = open(artifact_manager.getDir() + file_path)
    data = json.load(json_file)

    metric = []
    epoch = []

    for entry in data['entries']:
        if entry['phase'] != phase:
            continue
        metric.append(entry[metric_key])
        epoch.append(entry['epoch'])

    return metric, epoch

class Logger():
    def __init__(self, name):
        self.log = {}
        self.log['training_start'] = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.log['model_name'] = name
        self.log['entries'] = []

    def add(self, entry):
        self.log['training_update'] = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.log['entries'].append(entry)

        with open(artifact_manager.getDir() + self.log['model_name'] + '_training_log.json', 'w') as outfile:
            json.dump(self.log, outfile, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Checkpointer():
    def __init__(self, checkpoint_type):
        self.checkpoint_type = checkpoint_type
    
    # event consumer
    def event(self, event):
        map = {
            'epoch': self.checkpoint_epoch
        }

        map[self.checkpoint_type](event)

    # checkpoint types
    def checkpoint_epoch(self, event):
        if event['name'] == "EpochEnd":
            event["model"].save(tag="_epoch_" + str(event["epoch"]))

class Visualizer():
    def __init__(self):
        pass
    # event consumer
    def event(self, event):
        if event['name'] == "EpochEnd":
            from data.voc2012_loader_segmentation import PascalVOCSegmentation
            from visualize.visualize import visualize
            from measure.measure import measure
            from torch.utils.data.dataloader import DataLoader
            from artifacts.artifact_manager import artifact_manager

            dataloader = DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=4)

            model = event["model"]
            output_v = event["model"].name + '_visualization_epoch_' + str(event["epoch"]) +'/'
            output_m = event["model"].name + '_visualization_epoch_' + str(event["epoch"]) + '_measure'

            visualize(model, dataloader, output_v, max_count=32*8)
            measure(output_v, output_m)

class Schedule():
    def __init__(self, step_init, step_min, step_max, val_min, val_max):
        self.step_curr = step_init
        self.step_min = step_min
        self.step_max = step_max
        self.val_min = val_min
        self.val_max = val_max

    def step(self):
        self.step_curr += 1

    def get_val(self):
        if self.step_curr < self.step_min:
            return self.val_min
        if self.step_curr > self.step_max:
            return self.val_max
        
        norm_progress = (self.step_curr - self.step_min) / (self.step_max - self.step_min)
        return (self.val_max - self.val_min) * norm_progress + self.val_min
