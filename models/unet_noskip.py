
import cv2
import numpy as np
import torch, torchvision
import os
from data.voc2012 import label_to_image

from models.model_base import ModelBase

from metrics.accuracy import accuracy

def double_conv(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


class UNetNoSkip(ModelBase):
    def __init__(self, **kwargs):
        super(UNetNoSkip, self).__init__(**kwargs)
        
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
        
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)
        self.conv_comb = torch.nn.Conv2d(64, 20, 3, padding=1)
        self.conv_comb1 = torch.nn.Conv2d(20, 16, 3, padding=1)
        self.conv_comb2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv_comb3 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv_last = torch.nn.Conv2d(16, kwargs['outputs'], 1)

        self.gmp = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.loss_bce_func = torch.nn.BCELoss()

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, inputs):
        x = inputs['image']
        input = x.clone().detach().cpu().numpy()

        x = self.vgg16.features(x)

        x = self.upsample(x)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = self.dconv_up1(x)

        x = self.upsample(x)
        x = self.conv_comb(x)
        x = self.sigmoid(x)

        comb = self.gmp(x)
        comb = torch.flatten(comb, 1)
        self.loss_bce = self.loss_bce_func(comb, inputs['classification'])

        comb = x.clone().detach().cpu().numpy()
        comb = comb[0]
        cv2.imshow('comb0', comb[0])
        cv2.imshow('comb1', comb[1])
        cv2.imshow('comb2', comb[2])
        cv2.imshow('comb3', comb[3])
        cv2.imshow('comb4', comb[4])
        cv2.imshow('comb5', comb[5])

        
        x = self.conv_comb1(x)
        x = self.conv_comb2(x)
        x = self.conv_comb3(x)

        x = self.conv_last(x)
        x = self.sigmoid(x)

        input = input[0]
        input = np.moveaxis(input, 0, 2)
        
        output = x.clone().detach().cpu().numpy()
        output = output[0]
        output = np.moveaxis(output, 0, 2)

        cv2.imshow('input', input)
        cv2.imshow('output', output)
        cv2.waitKey(1)

        outputs = {
            'reconstruction': x
        }

        return outputs

    def backward(self, outputs, labels):
        if self.training:
            loss = self.loss_function(outputs['reconstruction'], labels['reconstruction'])
            loss.backward(retain_graph=True)
            self.loss_bce.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def metrics(self, outputs, labels):
        metrics = {
            'reconstruction': {
                'accuracy': accuracy,
            }
        }
        metrics_output = {}
        for output_key in metrics:
            metrics_output[output_key] = {}
            for metric_name in metrics[output_key]:
                metric_func = metrics[output_key][metric_name]
                metric_result = metric_func(outputs[output_key].cpu().detach().numpy(), labels[output_key].cpu().detach().numpy())
                metrics_output[output_key][metric_name] = metric_result

        return metrics_output

    def should_save(self, metrics_best, metrics_last):
        metric_best = metrics_best['reconstruction']['accuracy']
        metric_last = metrics_last['reconstruction']['accuracy']
        return metric_last >= metric_best

    def segment(self, images, class_labels):
        return 0