
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

        vgg = torchvision.models.vgg16(pretrained=True, progress=True)
        vgg.avgpool = None
        vgg.classifier = None
        vgg.features = vgg.features[:-1]
        self.shared_features = vgg.features

        # generator_vgg = torchvision.models.vgg16(pretrained=True, progress=True)
        # generator_vgg.avgpool = None
        # generator_vgg.classifier = None
        # generator_vgg.features = generator_vgg.features[:-1]
        # self.generator_features = generator_vgg.features
        self.generator_upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.generator_bottle = torch.nn.Conv2d(512, 20, 1)
        self.generator_dconv3 = double_conv(20, 256)
        self.generator_dconv2 = double_conv(256, 128)
        self.generator_dconv1 = double_conv(128, 64)
        self.generator_dconv0 = torch.nn.Conv2d(64, 3, 1)
        self.generator_sigmoid = torch.nn.Sigmoid()

        # self.classifier_conv = torch.nn.Conv2d(512, 20, 1)
        # self.classifier_gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.classifier_sigmoid = torch.nn.Sigmoid()


        # discriminator_vgg = torchvision.models.vgg16(pretrained=True, progress=True)
        # discriminator_vgg.avgpool = None
        # discriminator_vgg.classifier = None
        # discriminator_vgg.features = discriminator_vgg.features[:-1]
        # self.discriminator_features = discriminator_vgg.features
        # self.discriminator_conv = torch.nn.Conv2d(512, 1, 1)
        # self.discriminator_gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.discriminator_sigmoid = torch.nn.Sigmoid()
        # self.discriminator_loss = torch.nn.BCELoss()


        # # Unfreeze last conv layer
        # total = 0
        # count = 0
        # unfreeze = 2
        # for param in self.vgg16.parameters():
        #     total += 1
        # for param in self.vgg16.parameters():
        #     if (count >= total-unfreeze*2):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        #     count += 1

    # def discriminate(self, inputs, labels):
    #     discrimination = inputs['image']
    #     discrimination = self.discriminator_features(discrimination)
    #     discrimination = self.discriminator_conv(discrimination)
    #     discrimination = self.discriminator_gap(discrimination)
    #     discrimination = self.discriminator_sigmoid(discrimination)
    #     return discrimination

    # def classify(self, inputs):
    #     classification = self.classifier_conv(inputs)
    #     classification = self.classifier_gap(classification)
    #     classification = torch.flatten(classification, 1)
    #     classification = self.classifier_sigmoid(classification)
    #     return classification

    def generate(self, inputs):
        #reconstruction = inputs['image']
        #reconstruction = self.generator_features(reconstruction)

        reconstruction = self.generator_bottle(inputs)

        reconstruction = self.generator_upsample(reconstruction)
        reconstruction = self.generator_dconv3(reconstruction)

        reconstruction = self.generator_upsample(reconstruction)
        reconstruction = self.generator_dconv2(reconstruction)

        reconstruction = self.generator_upsample(reconstruction)
        reconstruction = self.generator_dconv1(reconstruction)

        reconstruction = self.generator_upsample(reconstruction)
        reconstruction = self.generator_dconv0(reconstruction)

        reconstruction = self.generator_sigmoid(reconstruction)
        return reconstruction
        
    def forward(self, inputs):
        original = inputs['image']
        features = self.shared_features(original)

        reconstruction = self.generate(features)
        # classification = self.classify(features)

        # maxed_values, maxed_indices = torch.max(classification, 1)
        # selected_weight = self.classifier_conv.weight[maxed_indices[0]]

        # weighted_features = features * selected_weight
        # weighted_reconstruction = self.generate(weighted_features)

        # weighted_output = weighted_reconstruction.clone().detach().cpu().numpy()
        # weighted_output = weighted_output[0]
        # weighted_output = np.moveaxis(weighted_output, 0, -1)
        # cv2.imshow('weighted_output', weighted_output)

        input = original.clone().detach().cpu().numpy()
        input = input[0]
        input = np.moveaxis(input, 0, -1)

        output = reconstruction.clone().detach().cpu().numpy()
        output = output[0]
        output = np.moveaxis(output, 0, -1)

        

        cv2.imshow('input', input)
        cv2.imshow('output', output)
        
        cv2.waitKey(1)

        outputs = {
            # 'classification': classification,
            'reconstruction': reconstruction
        }

        return outputs

    def load(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(package_directory, 'checkpoints', self.name + '.pt')
        self.load_state_dict(torch.load(weight_path))

    def save(self):
        print('saving model')
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(package_directory, 'checkpoints', self.name + '.pt')
        torch.save(self.state_dict(), weight_path)
