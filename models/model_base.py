import torch
from artifacts.artifact_manager import artifact_manager

class ModelBase(torch.nn.Module):
    def __init__(self, name='model_base', event_consumers=[], **kwargs):
        super().__init__()
        self.name = name
        self.event_consumers = event_consumers
        self.metrics_schema = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, *args):
        raise Exception('No forward pass specified')

    def backward(self, *args):
        raise Exception('No backward pass specified')

    def metrics(self, outputs, labels):
        metrics_output = {}
        for output_key in self.metrics_schema:
            metrics_output[output_key] = {}
            for metric_name in self.metrics_schema[output_key]:
                metric_func = self.metrics_schema[output_key][metric_name]
                metric_result = metric_func(outputs[output_key].cpu().detach().numpy(), labels[output_key].cpu().detach().numpy())
                metrics_output[output_key][metric_name] = metric_result

        return metrics_output

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