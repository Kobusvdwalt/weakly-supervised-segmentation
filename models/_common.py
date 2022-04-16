import torchvision
import torch

from artifacts.artifact_manager import artifact_manager
from training._common import Logger

def build_vgg_features(pretrained=True, unfreeze_from=10):
    vgg = torchvision.models.vgg16(pretrained=pretrained, progress=True)
    vgg.avgpool = None
    vgg.classifier = None
    vgg.features = vgg.features[:-1]

    # if padding == True:
    #     for feature_index, feature in enumerate(vgg.features):
            # print(feature_index, type(feature))
            # if type(feature) is torch.nn.modules.conv.Conv2d:
            #     newFeature = torch.nn.Conv2d(feature.in_channels, feature.out_channels, feature.kernel_size, padding=1)
            #     vgg.features[feature_index] = newFeature
            #     print('true')
            # if type(feature) is torch.nn.modules.pooling.MaxPool2d:
            #     vgg.features[feature_index] = torch.nn.MaxPool2d(feature.kernel_size, feature.stride, padding=1)


    count = 0
    for param in vgg.parameters():
        count += 1
        if count <= 2 * unfreeze_from:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return vgg.features

def print_params(params, name):
    print(name + ' paramaters:')
    for param_index, param in enumerate(params):
        print(param_index, param.requires_grad, param.shape)

def ff(f):
    return "{:.4f}".format(f)
def fi(i):
    return "{:4d}".format(i)

class ModelBase(torch.nn.Module):
    def __init__(self, name='model_base', event_consumers=[], **kwargs):
        super().__init__()
        self.name = name
        self.event_consumers = event_consumers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = Logger(self.name)

    def event(self, event):
        event['model'] = self
        for consumer in self.event_consumers:
            consumer.event(event)

    def load(self, tag=""):
        weight_path = artifact_manager.getDir() + self.name + "_weights" + tag + ".pt"
        # print(self.state_dict().keys())
        # files = torch.load(weight_path)
        # print(files.keys())
        self.load_state_dict(torch.load(weight_path))

    def save(self, tag=""):
        weight_path = artifact_manager.getDir() + self.name + "_weights" + tag + ".pt"
        torch.save(self.state_dict(), weight_path)
        print('Saved Model: ', weight_path)

    def new_instance(self):
        raise Exception("new_instance method not implemented")