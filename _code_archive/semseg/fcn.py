import torchvision
import torch

class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
        print (self.fcn)

    def forward(self, x):
        x = self.fcn(x)
        return x

fcn = FCN()
# deeplab.load_state_dict(torch.load('checkpoints/vgg_cam.pt'))