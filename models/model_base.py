import torch, os
from artifacts import artifact_manager
class ModelBase(torch.nn.Module):
    def __init__(self, name='model_base', **kwargs):
        super().__init__()
        self.name = name

    def forward(self, *args):
        raise Exception('No forward pass specified')

    def backward(self, *args):
        raise Exception('No backward pass specified')

    def metrics(self, *args):
        raise Exception('No metrics pass specified')

    def should_save(self):
        raise Exception('No should_save pass specified')

    def epoch_start(self):
        i = 0

    def load(self):
        weight_path = artifact_manager.instance.getArtifactDir() + self.name + "_weights.pt"
        self.load_state_dict(torch.load(weight_path))

    def save(self):
        print('saving model')
        weight_path = artifact_manager.instance.getArtifactDir() + self.name + "_weights.pt"
        torch.save(self.state_dict(), weight_path)