import torchvision
import torch
import os
class Vgg16GAP(torch.nn.Module):
    def __init__(self, name, outputs):
        super(Vgg16GAP, self).__init__()
        self.name = name + '_vgg16_gap_unfreeze_0'
        print(self.name)
        self.vgg = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg.features = self.vgg.features[:-1]
        self.vgg.avgpool = None
        self.vgg.classifier = None

        # Unfreeze last conv layer
        total = 0
        count = 0
        unfreeze = 2
        for param in self.vgg.parameters():
            total += 1
        for param in self.vgg.parameters():
            if (count >= total-unfreeze*2):
                print('layer unfrozen' + str(count))
                param.requires_grad = True
            else:
                param.requires_grad = False
            count += 1
        
        self.conv = torch.nn.Conv2d(512, outputs, 1)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.vgg.features(x)
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
