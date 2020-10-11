import torchvision
import torch
import os
import numpy as np
import cv2

class Vgg16GMP(torch.nn.Module):
    def __init__(self, name, outputs):
        super(Vgg16GMP, self).__init__()
        self.name = name + '_vgg16_gmp'
        print(self.name)
        self.vgg = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg.features = self.vgg.features[:-1]
        self.vgg.avgpool = None
        self.vgg.classifier = None

        # Unfreeze last conv layer
        total = 0
        count = 0
        for param in self.vgg.parameters():
            total += 1
        for param in self.vgg.parameters():
            if (count > total-6):
                param.requires_grad = True
            else:
                param.requires_grad = False
            count += 1
        
        print(total)

        self.conv = torch.nn.Conv2d(512, outputs, 1)
        self.gmp = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        input_np = x.clone().detach().cpu().numpy()

        x = self.vgg.features(x)
        x = self.conv(x)
        
        features_np = x.clone().detach().cpu().numpy()

        x = self.gmp(x)
        x = torch.flatten(x, 1)
        x = self.sigmoid(x)

        # output_np = x.clone().detach().cpu().numpy()
        # output_np = np.moveaxis(output_np, 1, -1)
        # cv2.imshow('output', output_np[0])

        # Visualize Input
        input_np = np.moveaxis(input_np, 1, -1)        
        cv2.imshow('input', input_np[0])

        # Visualize Features
        count = features_np[0].shape[0]
        instance = 0
        
        stacked_w = None
        while(instance < count-1):
            stacked_h = features_np[0, instance]
            instance += 1
            for i in range(0, 3):
                stacked_h = np.concatenate((stacked_h, features_np[0, instance]), axis=0)
                instance += 1

            if stacked_w is None:
                stacked_w = stacked_h
            else:
                stacked_w = np.concatenate((stacked_w, stacked_h), axis=1)

        stacked_w = cv2.resize(stacked_w, (stacked_w.shape[1] * 16, stacked_w.shape[0] * 16), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('features', stacked_w)
        cv2.waitKey(0)

        # Return output
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
