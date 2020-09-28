import torchvision
import torch
import os

# torch.utils.model_zoo.load_url( os.path.abspath('./checkpoints') )
# C:\Users\Kobus\.cache\torch\checkpoints
class DeepLab101(torch.nn.Module):
    def __init__(self, name, outputs):
        super(DeepLab101, self).__init__()
        self.name = name
        self.deeplab101 = torchvision.models.segmentation.deeplabv3_resnet101 (pretrained=True, progress=True)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.deeplab101(x)['out']
        x = self.sigmoid(x)
        return x

    def load(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(package_directory, 'checkpoints', self.name + '.pt')
        self.load_state_dict(torch.load(weight_path))

    def save(self):
        print('saving model')
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(package_directory, 'checkpoints', self.name + '.pt')
        torch.save(self.state_dict(), weight_path)
