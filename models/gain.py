
from data.voc2012_loader_classification import classification_labels
from data.voc2012 import label_to_image
import cv2
import numpy as np
import torch, torchvision
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
class Gain(ModelBase):
    def __init__(self, **kwargs):
        super(Gain, self).__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.features = build_vgg_features()
        self.conv_comb = torch.nn.Conv2d(512, 21, 1)
        self.flatten = torch.nn.Flatten(1, -1)

        self.sigmoid = torch.nn.Sigmoid()

        self.gmp = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.ups = torch.nn.Upsample(scale_factor=16, mode='nearest') # , align_corners=True
        self.step = 0
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.metrics_schema = {
            'classification': {
                'f1': f1,
            }
        }

        print_params(self.parameters(), "Transformer")

    def segment_pass(self, image):
        segmentation = self.features(image)
        segmentation = self.conv_comb(segmentation)
        segmentation = self.sigmoid(segmentation)
        segmentation = self.ups(segmentation)

        return segmentation

    def build_label(self, transformer):
        transformer_vis = transformer.clone().detach().cpu().numpy()
        label_vis = label_to_image(transformer_vis)
        return label_vis

    def forward(self, inputs):
        image = inputs['image']
        classification_label = inputs['label']

        # First pass
        segmentation = self.segment_pass(image)
        classification = torch.flatten(self.gmp(segmentation[:, 1:]), 1, 3)
        loss_bce = self.loss_bce(classification, classification_label)

        # Mask generation
        segmentation = segmentation.clone()
        segmentation[:, 1:] *= classification_label.unsqueeze(-1).unsqueeze(-1)
        mask, _ = torch.max(segmentation[:, 1:], dim=1, keepdim=True)
        segmentation[:, 0] = 1 - mask[:, 0]
        transformed = image * (1 - mask) + 0.5 * mask

        # Second pass
        # segmentation_s = self.segment_pass(transformed)
        # classification_s = torch.flatten(self.gmp(segmentation_s[:, 1:]), 1, 3)
        # loss_atm = torch.mean(classification_s[classification_label > 0.5])
            

        if self.training:
            self.step += 1
            loss = loss_bce # + loss_atm * 0.1
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

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