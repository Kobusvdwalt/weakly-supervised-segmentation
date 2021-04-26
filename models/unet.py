
import cv2
import numpy as np
import torch, torchvision
import os
from data.voc2012 import label_to_image

from models.model_base import ModelBase

from metrics.iou import iou, class_iou

def double_conv(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


class UNet(ModelBase):
    def __init__(self, **kwargs):
        super(UNet, self).__init__(**kwargs)
        
        self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg16.features = self.vgg16.features[:-1]
        self.vgg16.avgpool = None
        self.vgg16.classifier = None

        # Unfreeze conv layers
        total = 0
        count = 0
        unfreeze = 2
        for param in self.vgg16.parameters():
            total += 1
        for param in self.vgg16.parameters():
            if (count >= total-unfreeze*2):
                param.requires_grad = True
            else:
                param.requires_grad = False
            count += 1

        self.sigmoid = torch.nn.Sigmoid()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(512 + 512, 256)
        self.dconv_up2 = double_conv(256 + 256, 128)
        self.dconv_up1 = double_conv(128 + 128, 64)
        self.conv_comb = torch.nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.conv_last = torch.nn.Conv2d(64, kwargs['outputs'], 1)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.vgg16.features[3].register_forward_hook(output_hook)
        self.vgg16.features[8].register_forward_hook(output_hook)
        self.vgg16.features[15].register_forward_hook(output_hook)
        self.vgg16.features[22].register_forward_hook(output_hook)

        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.metrics_schema = {
            'segmentation': {
                'miou': iou,
                '_class_iou': class_iou
            }
        }

    def forward(self, inputs):
        x = inputs['image']

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
        x = self.conv_comb(x)
        x = self.conv_last(x)
        x = self.sigmoid(x)

        self.intermediate_outputs.clear()

        outputs = {
            'segmentation': x
        }

        return outputs

    def backward(self, outputs, labels):
        if self.training:
            loss = self.loss_function(outputs['segmentation'], labels['segmentation'])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def should_save(self, metrics_best, metrics_last):
        metric_best = metrics_best['segmentation']['miou']
        metric_last = metrics_last['segmentation']['miou']
        return metric_last >= metric_best

    def segment(self, images, class_labels):
        x = images
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
        x = self.conv_comb(x)
        x = self.conv_last(x)
        x = self.sigmoid(x)

        # Build label
        result = np.zeros((images.shape[0], images.shape[2], images.shape[3], 3))

        for batch_index in range(0, images.shape[0]):
            output = x.clone().detach().cpu().numpy()
            result[batch_index] = label_to_image(output[batch_index])

        self.intermediate_outputs.clear()

        return result