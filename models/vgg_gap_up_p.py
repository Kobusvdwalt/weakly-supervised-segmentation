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

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = torch.nn.Conv2d(512 + 512, 256, 1)
        self.conv2 = torch.nn.Conv2d(256 + 256, 128, 1)

        self.conv = torch.nn.Conv2d(128, outputs, 1)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Conv 1
        conv1 = self.vgg16.features[0](x)
        conv1 = self.vgg16.features[1](conv1)

        conv1 = self.vgg16.features[2](conv1)
        conv1 = self.vgg16.features[3](conv1)

        # Conv 2
        conv2 = self.vgg16.features[4](conv1)

        conv2 = self.vgg16.features[5](conv2)
        conv2 = self.vgg16.features[6](conv2)

        conv2 = self.vgg16.features[7](conv2)
        conv2 = self.vgg16.features[8](conv2)

        # Conv 3
        conv3 = self.vgg16.features[9](conv2)

        conv3 = self.vgg16.features[10](conv3)
        conv3 = self.vgg16.features[11](conv3)

        conv3 = self.vgg16.features[12](conv3)
        conv3 = self.vgg16.features[13](conv3)

        conv3 = self.vgg16.features[14](conv3)
        conv3 = self.vgg16.features[15](conv3)

        # Conv 4
        conv4 = self.vgg16.features[16](conv3)

        conv4 = self.vgg16.features[17](conv4)
        conv4 = self.vgg16.features[18](conv4)

        conv4 = self.vgg16.features[19](conv4)
        conv4 = self.vgg16.features[20](conv4)

        conv4 = self.vgg16.features[21](conv4)
        conv4 = self.vgg16.features[22](conv4)

        # Conv 5
        conv5 = self.vgg16.features[23](conv4)

        conv5 = self.vgg16.features[24](conv5)
        conv5 = self.vgg16.features[25](conv5)

        conv5 = self.vgg16.features[26](conv5)
        conv5 = self.vgg16.features[27](conv5)

        conv5 = self.vgg16.features[28](conv5)
        conv5 = self.vgg16.features[29](conv5)

        x = self.upsample(conv5)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv1(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv2(x)

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
