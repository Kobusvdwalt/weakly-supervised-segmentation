
from data.voc2012 import label_to_image
import cv2
import numpy as np
import torch
import random
from models._common import ModelBase
from metrics.f1 import f1
from data.voc2012_loader_segmentation import PascalVOCSegmentation

from models._common import build_vgg_features, print_params

##################################################################################################################
# Classifier
##################################################################################################################
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            build_vgg_features(),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Linear(512, 48),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Linear(48, 20),
            torch.nn.Sigmoid()
        )
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        print_params(self.parameters(), "Classifier")

    def forward(self, image):
        return self.classifier(image)

##################################################################################################################
# Transformer
##################################################################################################################
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.features = build_vgg_features()
        self.dconv_up3 = self.double_conv(512, 256, 3, 1)
        self.dconv_up2 = self.double_conv(256, 128, 3, 1)
        self.dconv_up1 = self.double_conv(128, 64, 3, 1)
        self.conv_comb = torch.nn.Conv2d(64, 21, 3, padding=1)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.features[3].register_forward_hook(output_hook)
        self.features[8].register_forward_hook(output_hook)
        self.features[15].register_forward_hook(output_hook)
        self.features[22].register_forward_hook(output_hook)

        self.sigmoid = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten(1, -1)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.gmp = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        print_params(self.parameters(), "Transformer")

    def double_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
        )

    def segment(self, image):
        transformer = self.features(image)

        transformer = self.upsample(transformer)
        transformer += self.intermediate_outputs[3]
        transformer = self.dconv_up3(transformer)

        transformer = self.upsample(transformer)
        transformer += self.intermediate_outputs[2]
        transformer = self.dconv_up2(transformer)

        transformer = self.upsample(transformer)
        transformer += self.intermediate_outputs[1]
        transformer = self.dconv_up1(transformer)

        transformer = self.upsample(transformer)
        transformer += self.intermediate_outputs[0]
        transformer = self.conv_comb(transformer)

        transformer = self.sigmoid(transformer)

        classification = self.gmp(transformer[:, 1:])
        classification = self.flatten(classification)

        self.intermediate_outputs.clear()

        return transformer, classification

    def build_label(self, transformer):
        transformer_vis = transformer.clone().detach().cpu().numpy()
        label_vis = label_to_image(transformer_vis)
        return label_vis

    def forward(self, images, classification_labels):
        segmentation, classification = self.segment(images)

        # Mask generation
        segmentation = segmentation.clone()
        segmentation[:, 0] = 0.51
        segmentation[:, 1:] *= classification_labels.unsqueeze(-1).unsqueeze(-1)
        segmentation = torch.sigmoid((segmentation -0.5) * 100)

        return {
            'classification' : classification,
            'segmentation' : segmentation,
        }

##################################################################################################################
# Super model
##################################################################################################################

class WASS(ModelBase):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.step = -1
        self.step_count = 0

        self.classifier = Classifier()
        self.transformer = Transformer()
        self.segmentation_loader = PascalVOCSegmentation('train')

        self.metrics_schema = {
            'classification': {
                'f1': f1,
            }
        }

    def forward(self, inputs):
        # Actual forward pass
        image = inputs['image']
        classification_label = inputs['label']

        epoch_size = 715
        interval_size = 32

        # Adverserial training controller
        if self.training:
            if self.step_count // interval_size % 2 == 0:
                cv2.imshow('image', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
                with torch.no_grad():
                    tp = self.transformer(image, classification_label)
                    segmentation = tp['segmentation']
                    mask, _ = torch.max(segmentation[:, 1:], dim=1, keepdim=True)

                    if random.random() > 0.2:
                        image = image * (1 - mask)

                cv2.imshow('segmentation', self.transformer.build_label(segmentation[0]))
                cv2.imshow('erased', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
                cv2.waitKey(1)

                # Train Classifier
                classification = self.classifier(image)
                loss_bce_class = self.classifier.loss_bce(classification, classification_label)
                loss_bce_class.backward()
                self.classifier.optimizer.step()

                
            else:
                # Train Transformer
                tp = self.transformer(image, classification_label)
                segmentation = tp['segmentation']
                classification = tp['classification']

                cv2.imshow('segmentation', self.transformer.build_label(segmentation[0]))
                cv2.imshow('image', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
                cv2.waitKey(1)

                loss_transformer = self.transformer.loss_bce(classification, classification_label)
                loss_mask = torch.tensor(0, dtype=torch.float, device=self.device, requires_grad=True)
                loss_mining = torch.tensor(0, dtype=torch.float, device=self.device, requires_grad=True)

                segmentation = segmentation[:, 1:]
                for image_no in range(image.shape[0]):
                    for mask_no in range(segmentation.shape[1]):
                        if classification_label[image_no, mask_no] > 0.5:
                            mask = segmentation[image_no, mask_no]
                            erased = image[image_no] * (1 - mask)
                            erased_c = self.classifier(erased.unsqueeze(0))

                            loss_mining = loss_mining.clone() + erased_c[0, mask_no]
                            loss_mask = loss_mask.clone() + torch.mean(mask)

                            if image_no == 0:
                                cv2.imshow('mask', mask.clone().detach().cpu().numpy())
                                cv2.imshow('erased', np.moveaxis(erased.clone().detach().cpu().numpy(), 0, -1))
                                cv2.waitKey(1)

                loss_final = loss_transformer + loss_mining + loss_mask * 0.2
                loss_final.backward()
                self.transformer.optimizer.step()

            self.classifier.optimizer.zero_grad()
            self.transformer.optimizer.zero_grad()

            self.step_count += 1
        else:
            with torch.no_grad():
                classification = self.classifier(image)

        outputs = {
            'classification': classification,
        }

        return outputs

    def backward(self, outputs, labels):
        i = 0

    def segment(self, images, class_labels):
        x_clean = self.transformer.segment(images)

        # Build label
        result = np.zeros(images.shape)
        result = np.moveaxis(result, 1, -1)
        for i in range(0, images.shape[0]):
            result[i] = self.transformer.build_label(x_clean[i])
        return result

    def should_save(self, metrics_best, metrics_last):
        return True



##################################################################################################################
# Mask Discriminator
##################################################################################################################
# class Discriminator(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.discriminator = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.AdaptiveMaxPool2d(output_size=(1, 1)),
#             torch.nn.Flatten(1, 3),
#             torch.nn.Linear(128, 32),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.Linear(32, 1),
#             torch.nn.Sigmoid()
#         )
#         self.loss_bce = torch.nn.BCELoss()
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

#     def forward(self, image):
#         return self.discriminator(image)


# Train Discriminator
                # discrimination_input = mask.clone()
                # discrimination_label = np.full((mask.shape[0], 1), 0.1)

                # for i in range(0, mask.shape[0]):
                #     if random.random() < 0.5:
                #         continue
                #     else:
                #         _, label_dict, _ = self.segmentation_loader.__getitem__(random.randint(0, self.segmentation_loader.__len__() -1))
                #         seg = label_dict['segmentation']
                #         seg = np.max(seg[1:], axis=0)
                #         discrimination_input[i] = torch.tensor(seg, device=self.device, dtype=torch.float).unsqueeze(0)
                #         discrimination_label[i] = 0.9
                #         if random.random() < 0.05:
                #             discrimination_label[i] = 0.1

                # discrimination_input -= torch.rand(discrimination_input.shape, device=self.device) * 0.5
                # discrimination_input[discrimination_input > 0.9] = 0.9
                # discrimination_input[discrimination_input < 0.1] = 0.1

                # discrimination_label = torch.tensor(discrimination_label, device=self.device, dtype=torch.float)
                
                # cv2.imshow('disc_mask', discrimination_input[0, 0].clone().detach().cpu().numpy())
                # cv2.waitKey(1)

                # discrimination = self.discriminator(discrimination_input)
                # loss_bce_disc = self.discriminator.loss_bce(discrimination, discrimination_label)
                # loss_bce_disc.backward()
                # self.discriminator.optimizer.step()



