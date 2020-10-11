
import cv2
import numpy as np
import torch, torchvision
import os

def double_conv(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


class UNet(torch.nn.Module):
    def __init__(self, name, outputs):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg16.features = self.vgg16.features[:-1]
        self.vgg16.avgpool = None
        self.vgg16.classifier = None

        # Unfreeze all conv layers
        count = 0
        for param in self.vgg16.parameters():
            if (count > 22):
                param.requires_grad = True
            else:
                param.requires_grad = False
            count += 1

        self.conv = torch.nn.Conv2d(512, 20, 1)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(512, 256) # + 512
        self.dconv_up2 = double_conv(256, 128) # + 256
        self.dconv_up1 = double_conv(128, 64) # + 64
        self.conv_last = torch.nn.Conv2d(64, outputs, 1)
        
    def forward(self, x):
        x = self.vgg16.features(x)
        classifier = self.conv(x)

        x = self.upsample(x)
        x = self.upsample(x)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = self.dconv_up2(x)

        x = self.upsample(x)        
        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.sigmoid(x)

        # Visualize Latent
        latent_np = classifier.clone().detach().cpu().numpy()
        classifier = self.gap(classifier)
        classifier = torch.flatten(classifier, 1)
        classifier = self.sigmoid(classifier)
        count = latent_np[0].shape[0]
        instance = 0
        
        stacked_w = None
        while(instance < count-1):
            stacked_h = latent_np[0, instance]
            instance += 1
            for i in range(0, 3):
                stacked_h = np.concatenate((stacked_h, latent_np[0, instance]), axis=0)
                instance += 1

            if stacked_w is None:
                stacked_w = stacked_h
            else:
                stacked_w = np.concatenate((stacked_w, stacked_h), axis=1)

        stacked_w = cv2.resize(stacked_w, (stacked_w.shape[1] * 16, stacked_w.shape[0] * 16), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('latent', stacked_w)

        feature = latent_np[0, torch.argmax(classifier[0])]
        feature = cv2.resize(feature, (feature.shape[1] * 16,   feature.shape[0] * 16), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('feature', feature)

        return x, classifier

    def load(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(package_directory, 'checkpoints', self.name + '.pt')
        self.load_state_dict(torch.load(weight_path))

    def save(self):
        print('saving model')
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(package_directory, 'checkpoints', self.name + '.pt')
        torch.save(self.state_dict(), weight_path)
