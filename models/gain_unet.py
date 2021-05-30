
from data.voc2012_loader_classification import classification_labels
from data.voc2012 import label_to_image
import cv2
import numpy as np
import torch, torchvision
import os
import random
from models._common import ModelBase
from metrics.f1 import f1
from data.voc2012_loader_segmentation import PascalVOCSegmentation
from training._common import Schedule

def build_vgg_features():
    vgg = torchvision.models.vgg16(pretrained=True, progress=True)
    vgg.avgpool = None
    vgg.classifier = None
    vgg.features = vgg.features[:-1]
    count = 0
    for param in vgg.parameters():
        count += 1
        if count <= 2 * 10:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return vgg.features

def print_params(params, name):
    print(name)
    for param in params:
        print(type(param), param.size(), param.requires_grad)

##################################################################################################################
# UNET
##################################################################################################################
class Gain_UNET(ModelBase):
    def __init__(self, **kwargs):
        super(Gain_UNET, self).__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.features = build_vgg_features()
        self.dconv_up3 = self.double_conv(512, 256, 3, 1)
        self.dconv_up2 = self.double_conv(256, 128, 3, 1)
        self.dconv_up1 = self.double_conv(128, 64, 3, 1)
        self.conv_comb = torch.nn.Conv2d(64, 21, 1)
        self.flatten = torch.nn.Flatten(1, -1)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.features[3].register_forward_hook(output_hook)
        self.features[8].register_forward_hook(output_hook)
        self.features[15].register_forward_hook(output_hook)
        self.features[22].register_forward_hook(output_hook)

        self.sigmoid = torch.nn.Sigmoid()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.gmp = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.step = 0
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.metrics_schema = {
            'classification': {
                'f1': f1,
            }
        }

        print_params(self.parameters(), "Transformer")

    def double_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
        )

    def segment_pass(self, image):
        segmentation = self.features(image)

        segmentation = self.upsample(segmentation)
        segmentation += self.intermediate_outputs[3]
        segmentation = self.dconv_up3(segmentation)

        segmentation = self.upsample(segmentation)
        segmentation += self.intermediate_outputs[2]
        segmentation = self.dconv_up2(segmentation)

        segmentation = self.upsample(segmentation)
        segmentation += self.intermediate_outputs[1]
        segmentation = self.dconv_up1(segmentation)
        
        segmentation = self.upsample(segmentation)
        segmentation += self.intermediate_outputs[0]
        segmentation = self.conv_comb(segmentation)

        segmentation = self.sigmoid(segmentation)

        classification = self.gmp(segmentation[:, 1:])
        classification = self.flatten(classification)

        self.intermediate_outputs.clear()

        return segmentation, classification

    def build_label(self, transformer):
        transformer_vis = transformer.clone().detach().cpu().numpy()
        label_vis = label_to_image(transformer_vis)
        return label_vis

    def forward(self, inputs):
        image = inputs['image']
        classification_label = inputs['label']

        # First pass
        segmentation, classification = self.segment_pass(image)
        loss_bce = self.loss_bce(classification, classification_label)

        # Mask generation
        segmentation = segmentation.clone()
        segmentation[:, 1:] *= classification_label.unsqueeze(-1).unsqueeze(-1)
        mask, _ = torch.max(segmentation[:, 1:], dim=1, keepdim=True)
        segmentation[:, 0] = 1 - mask[:, 0]
        transformed = image * (1 - mask) + 0.5 * mask

        if self.training:
            self.step += 1
            loss = loss_bce
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        # for i in range(0, 20):
        #     d = transformer[0][i].clone().detach().cpu().numpy()
        #     cv2.imshow('f_' + str(i), d)

        cv2.imshow('transformer_lab', self.build_label(segmentation[0]))
        cv2.imshow('transformer_mas', mask[0, 0].clone().detach().cpu().numpy())
        cv2.imshow('transformer_inp', np.moveaxis(transformed[0].clone().detach().cpu().numpy(), 0, -1))
        cv2.waitKey(1)

        outputs = {
            'classification': classification,
        }

        return outputs


    def segment(self, images, class_labels):
        x_clean = self.transformer.segment(images)

        # Build label
        result = np.zeros(images.shape)
        result = np.moveaxis(result, 1, -1)
        for i in range(0, images.shape[0]):
            result[i] = self.transformer.build_label(x_clean[i])
        return result

    def backward(self, outputs, labels):
        i = 0

    def should_save(self, metrics_best, metrics_last):
        return True