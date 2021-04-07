import torch, json
import numpy as np

def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device).float()
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
        i = 0
    # event consumer
    def event(self, event):
        if event['name'] == "EpochEnd":
            from data.voc2012_loader_segmentation import PascalVOCSegmentation
            from visualize.visualize import visualize
            from measure.measure import measure
            from torch.utils.data.dataloader import DataLoader
            from artifacts.artifact_manager import artifact_manager

            dataloader = DataLoader(PascalVOCSegmentation('val'), batch_size=16, shuffle=False, num_workers=0)

            model = event["model"]
            output_v = event["model"].name + '_visualization_epoch_' + str(event["epoch"]) +'/'
            output_m = event["model"].name + '_visualization_epoch_' + str(event["epoch"]) + '_measure'

            visualize(model, dataloader, output_v)
            measure(output_v, output_m)