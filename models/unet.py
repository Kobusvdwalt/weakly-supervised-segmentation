import numpy as np
import torch, cv2
import wandb

from models._common import fi, ff
from data.voc2012 import label_to_image
from metrics.iou import class_iou, iou
from models._common import build_vgg_features
from models._common import ModelBase
from models._common import print_params

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

        self.metric_iou = 0
        self.metric_loss = 0

        self.features = build_vgg_features(unfreeze_from=0)
        self.dconv_up3 = self.double_conv(512, 256, 3, 1)
        self.dconv_up2 = self.double_conv(256, 128, 3, 1)
        self.dconv_up1 = self.double_conv(128, 64, 3, 1)
        self.dconv_up0 = self.double_conv(64, 32, 3, 1)
        self.conv_comb = torch.nn.Conv2d(32, 21, 1)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.features[3].register_forward_hook(output_hook)
        self.features[8].register_forward_hook(output_hook)
        self.features[15].register_forward_hook(output_hook)
        self.features[22].register_forward_hook(output_hook)

        self.loss_cce = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        print_params(self.parameters(), "UNET")

    def double_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
        )

    def forward(self, image):
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
        segmentation = self.dconv_up0(segmentation)
        segmentation = self.conv_comb(segmentation)

        self.intermediate_outputs.clear()

        return segmentation

    def event(self, event):
        if event['name'] == 'minibatch' and event['phase'] == 'train':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_cu = event['labels']['segmentation'].cuda(non_blocking=True)
            label_cu = torch.argmax(label_cu, 1).long()

            segmentation_result = self.forward(image_cu)

            loss = self.loss_cce(segmentation_result, label_cu)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            image = event['inputs']['image'].detach().numpy()
            image = image[0]
            image = np.moveaxis(image, 0, -1)
            cv2.imshow('image', image)

            label = event['labels']['segmentation'].detach().numpy()
            label_vis = label[0]
            label_vis = label_to_image(label_vis)
            cv2.imshow('label', label_vis)

            prediction = torch.softmax(segmentation_result.detach(), 1).cpu().numpy()
            prediction_vis = prediction[0]
            prediction_vis = label_to_image(prediction_vis)
            cv2.imshow('prediction', prediction_vis)

            cv2.waitKey(1)

            class_iou_mean = class_iou(prediction, label).mean()
            iou_result = iou(prediction, label)

            wandb.log({
                "class_iou_mean": class_iou_mean,
                "iou_result": iou_result,
                "loss": loss.detach().cpu().numpy(),
            })

        if event['name'] == 'epoch_end':
            print('')
            self.save()