import numpy as np
import torch, cv2

from models._common import fi, ff
from data.voc2012 import label_to_image
from metrics.iou import iou
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

        outputs = {
            'segmentation': segmentation
        }

        return outputs

    def event(self, event):
        if event['name'] == 'phase_start':
            if event['phase'] == 'train':
                self.train()
            else:
                self.eval()

        if event['name'] == 'minibatch' and event['phase'] == 'train':
            image = event['inputs']['image'].to(self.device)
            label = event['labels']['segmentation_cat'].to(self.device, non_blocking=True)
            segmentation_result = self.forward(image)

            loss_cce = self.loss_cce(segmentation_result['segmentation'], label)
            loss_cce.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.metric_loss += loss_cce.item()

            print(f'batch {fi(event["batch"])}', end='\r')

        if event['name'] == 'minibatch' and event['phase'] == 'val':
            with torch.no_grad():
                image = event['inputs']['image'].to(self.device)
                label = event['labels']['segmentation_cat'].to(self.device, non_blocking=True)
                segmentation_result = self.forward(image)

                loss_cce = self.loss_cce(segmentation_result['segmentation'], label)

                segmentation = torch.softmax(segmentation_result['segmentation'], dim=1).detach().cpu().numpy()
                self.metric_iou += iou(segmentation[:, 1:], event['labels']['segmentation'][:, 1:])
                self.metric_loss += loss_cce.item()

            cv2.imshow('image', np.moveaxis(event['inputs']['image'][0].numpy(), 0, -1))
            cv2.imshow('label', label_to_image(event['labels']['segmentation'][0]))
            cv2.imshow('output', label_to_image(segmentation[0]))
            cv2.waitKey(1)
            
            print(f'batch {fi(event["batch"])}', end='\r')

        if event['name'] == 'phase_end':
            batch_count = event['batch']
            print(f' epoch {fi(event["epoch"])} phase {event["phase"]} batch {fi(event["batch"])} miou {ff(self.metric_iou/batch_count)} loss {ff(self.metric_loss/batch_count)}')
            self.metric_iou = 0
            self.metric_loss = 0
