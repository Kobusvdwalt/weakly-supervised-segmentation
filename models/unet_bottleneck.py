
import cv2
import numpy as np
import torch, torchvision
import os
from data.voc2012 import label_to_image

def double_conv(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


class UNetBottleneck(torch.nn.Module):
    def __init__(self, name, outputs):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg16.features = self.vgg16.features[:-1]
        self.vgg16.avgpool = None
        self.vgg16.classifier = None

        # Unfreeze all conv layers
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.sigmoid = torch.nn.Sigmoid()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(512 + 512, 256)
        self.dconv_up2 = double_conv(256 + 256, 128)
        self.dconv_up1 = double_conv(128 + 128, 64)
        self.conv_1 = torch.nn.Conv2d(128, 3, 1)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.vgg16.features[3].register_forward_hook(output_hook)
        self.vgg16.features[8].register_forward_hook(output_hook)
        self.vgg16.features[15].register_forward_hook(output_hook)
        self.vgg16.features[22].register_forward_hook(output_hook)

        self.conv = torch.nn.Conv2d(512, outputs, 1)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        
    def forward(self, x):
        input = x.clone().detach().cpu().numpy()

        x = self.vgg16.features(x)

        x = self.upsample(x)
        x = torch.cat((x, self.intermediate_outputs[3]), dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat((x, self.intermediate_outputs[2]), dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat((x, self.intermediate_outputs[1]), dim=1)
        x = self.dconv_up1(x)

        x = self.upsample(x)
        x = torch.cat((x, self.intermediate_outputs[0]), dim=1)

        x = self.conv_1(x)
        
        x = self.sigmoid(x)
        output = x.clone().detach().cpu().numpy()

        x = self.vgg16.features(x)
        x = self.conv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.sigmoid(x)

        input = input[0]
        input = np.moveaxis(input, 0, -1)

        output = output[0]
        output = np.moveaxis(output, 0, -1)

        cv2.imshow('input', input)
        cv2.imshow('output', output)
        cv2.waitKey(1)
        
        self.intermediate_outputs.clear()
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
