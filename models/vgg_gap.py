import torchvision
import torch
import os

class Vgg16(torch.nn.Module):
    def __init__(self, name, outputs):
        super(Vgg16, self).__init__()
        self.name = name
        self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg16.features = self.vgg16.features[:-1]
        self.vgg16.avgpool = None
        self.vgg16.classifier = None

        # Unfreeze all conv layers
        count = 0
        for param in self.vgg16.parameters():
            if (count > 16):
                param.requires_grad = True
            else:
                param.requires_grad = False
            count += 1

        # self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.dense = torch.nn.Linear(512, outputs)
        # self.sigmoid = torch.nn.Sigmoid()

        self.conv = torch.nn.Conv2d(512, outputs, 1)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x = self.vgg16.features(x)
        # x = self.gap(x)
        # x = torch.flatten(x, 1)
        # x = self.dense(x)
        # x = self.sigmoid(x)

        x = self.vgg16.features(x)
        x = self.conv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
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
