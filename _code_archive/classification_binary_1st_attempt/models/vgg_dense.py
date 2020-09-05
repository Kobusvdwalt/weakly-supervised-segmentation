import torchvision
import torch
from data import classListArray

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)
        for param in self.vgg16.parameters():
            param.requires_grad = False
        self.vgg16.avgpool = None
        self.vgg16.classifier = None

        self.dense1 = torch.nn.Linear(8*8*512, 2048)
        self.relu1 = torch.nn.LeakyReLU(0.1)

        self.dense2 = torch.nn.Linear(2048, 1024)
        self.relu2 = torch.nn.LeakyReLU(0.1)

        self.dense3 = torch.nn.Linear(1024, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.vgg16.features(x)
        x = torch.flatten(x, 1)

        x = self.dense1(x)
        x = self.relu1(x)

        x = self.dense2(x)
        x = self.relu2(x)

        x = self.dense3(x)
        x = self.sigmoid(x)
        return x

vgg = Vgg16()
vgg.load_state_dict(torch.load('checkpoints/vgg.pt'))
vgg.eval()

print(vgg)