import cv2
import torchvision
import torch
import numpy as np

class Vgg16GAP(torch.nn.Module):
    def __init__(self, class_count=20):
        super().__init__()
        # Get a VGG16 network, pretrained on imagenet
        vgg = torchvision.models.vgg16(pretrained=True, progress=True)

        # Drop the linear layers
        vgg_features = vgg.features

        # Unfreeze last couple of convolutional layers
        for param_count, param in enumerate(vgg_features.parameters()):
            # There are two sets of params, weights and biases.
            # So we unfreeze 2 params per layer
            if param_count <= 2 * 6:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Bolt our feature extractor onto an output layer
        self.classifier = torch.nn.Sequential(
            vgg_features,
            torch.nn.Conv2d(512, class_count, 1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Sigmoid()
        )
    
        # Loss and optimizer
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_pass(self, inputs):
        return self.classifier(inputs)