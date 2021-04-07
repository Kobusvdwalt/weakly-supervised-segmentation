import torch
from artifacts.artifact_manager import artifact_manager

class ModelBase(torch.nn.Module):
    def __init__(self, name='model_base', event_consumers=[], **kwargs):
        super().__init__()
        self.name = name
        self.event_consumers = event_consumers

    def forward(self, *args):
        raise Exception('No forward pass specified')

    def backward(self, *args):
        raise Exception('No backward pass specified')

    def metrics(self, *args):
        raise Exception('No metrics pass specified')

    def should_save(self):
        raise Exception('No should_save pass specified')
    
    def event(self, event):
        event['model'] = self
        for consumer in self.event_consumers:
            consumer.event(event)

    def load(self, tag=""):
        weight_path = artifact_manager.getDir() + self.name + "_weights" + tag + ".pt"
        self.load_state_dict(torch.load(weight_path))

    def save(self, tag=""):
        weight_path = artifact_manager.getDir() + self.name + "_weights" + tag + ".pt"
        torch.save(self.state_dict(), weight_path)