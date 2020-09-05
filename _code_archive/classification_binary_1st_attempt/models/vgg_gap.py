import torchvision
import torch

from classification.data import classListArray

class Vgg16Binary(torch.nn.Module):
    def __init__(self):
        super(Vgg16Binary, self).__init__()

        self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)

        # Freeze all layers
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # Unreeze last 2 conv layers
        count = 0
        for param in self.vgg16.parameters():
            if count >= 18 and count <= 24:
                param.requires_grad = True
            count += 1

        self.vgg16.avgpool = None
        self.vgg16.classifier = None

        self.gap = torch.nn.MaxPool2d(8)

        self.dense1 = torch.nn.Linear(512, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = self.sigmoid(x)

        return x

vgg_binary = Vgg16Binary()
vgg_binary.load_state_dict(torch.load('./checkpoints/vgg_gap.pt'))