import torchvision
import torch

class DeepLab(torch.nn.Module):
    def __init__(self):
        super(DeepLab, self).__init__()
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        print (self.deeplab)

    def forward(self, x):
        x = self.deeplab(x)
        return x

model = DeepLab()
# deeplab.load_state_dict(torch.load('checkpoints/vgg_cam.pt'))