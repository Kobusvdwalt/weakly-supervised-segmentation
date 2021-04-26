
from data.voc2012 import label_to_image
import cv2
import numpy as np
import torch, torchvision
import os
import random
from models.model_base import ModelBase
from metrics.f1 import f1
from data.voc2012_loader_segmentation import PascalVOCSegmentation

def build_vgg_features():
    vgg = torchvision.models.vgg16(pretrained=True, progress=True)
    vgg.avgpool = None
    vgg.classifier = None
    vgg.features = vgg.features[:-1]
    for param in vgg.parameters():
        param.requires_grad = False

    return vgg.features

##################################################################################################################
# Mask Discriminator
##################################################################################################################
class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Linear(128, 32),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, image):
        real_fake = self.discriminator(image)
        return real_fake


##################################################################################################################
# Classifier
##################################################################################################################
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = build_vgg_features()
        self.conv_1_1 = torch.nn.Conv2d(512, 64, kernel_size=3, padding=1, dilation=1)
        self.conv_1_2 = torch.nn.Conv2d(512, 64, kernel_size=3, padding=2, dilation=2)
        self.conv_2 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.conv_3 = torch.nn.Conv2d(64, 20, 1)
        self.gmp = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = torch.nn.Sigmoid()

        self.loss_bce = torch.nn.BCELoss()
        print("Classifier params: ")
        for param in self.parameters():  
            print(type(param), param.size(), param.requires_grad)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, image):
        classification = self.features(image)

        classification_d1 = self.conv_1_1(classification)
        classification_d2 = self.conv_1_2(classification)
        classification = torch.cat((classification_d1, classification_d2), dim=1)

        classification = self.conv_2(self.relu(classification))
        classification = self.conv_3(self.relu(classification))

        classification = self.gmp(classification)
        classification = self.sigmoid(classification)
        classification = torch.flatten(classification, 1)

        return classification

##################################################################################################################
# Transformer
##################################################################################################################
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.features = build_vgg_features()
        self.dconv_up3 = self.double_conv(512 + 512, 128)
        self.dconv_up2 = self.double_conv(256 + 128, 64)
        self.dconv_up1 = self.double_conv(128 + 64, 32)
        self.conv_comb = torch.nn.Conv2d(64 + 32, 21, 3, padding=1)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.features[3].register_forward_hook(output_hook)
        self.features[8].register_forward_hook(output_hook)
        self.features[15].register_forward_hook(output_hook)
        self.features[22].register_forward_hook(output_hook)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.avg = torch.nn.AdaptiveAvgPool2d((16, 16))
        self.ups = torch.nn.Upsample(scale_factor=16, mode='nearest')
        self.gmp = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.loss_bce = torch.nn.BCELoss()
        print("Transformer params: ")
        for param in self.parameters():  
            print(type(param), param.size(), param.requires_grad)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def double_conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
        )

    def segment(self, image):
        # Unet start
        transformer = self.features(image)

        transformer = self.upsample(transformer)
        transformer = torch.cat((transformer, self.intermediate_outputs[3]), dim=1)
        transformer = self.dconv_up3(transformer)

        transformer = self.upsample(transformer)
        transformer = torch.cat((transformer, self.intermediate_outputs[2]), dim=1)
        transformer = self.dconv_up2(transformer)

        transformer = self.upsample(transformer)
        transformer = torch.cat((transformer, self.intermediate_outputs[1]), dim=1)
        transformer = self.dconv_up1(transformer)

        transformer = self.upsample(transformer)
        transformer = torch.cat((transformer, self.intermediate_outputs[0]), dim=1)
        transformer = self.conv_comb(transformer)
        transformer = self.tanh(transformer)
        transformer = self.sigmoid(10 * transformer)
        self.intermediate_outputs.clear()
        # Unet end

        return transformer

    def build_label(self, transformer):
        transformer_vis = transformer.clone().detach().cpu().numpy()
        label_vis = label_to_image(transformer_vis)
        return label_vis


    def forward(self, images, labels):
        mask_label = labels.unsqueeze(-1).unsqueeze(-1)
        mask_label[mask_label > 0.5] = 0.99
        mask_label[mask_label <= 0.5] = 0.01

        transformer = self.segment(images)
        transformer_c = torch.flatten(self.gmp(transformer[:, 1:]), 1)
        transformer = transformer.clone()

        transformer[:, 1:] *= mask_label
    
        transformer_m, _ = torch.max(transformer[:, 1:], dim=1, keepdim=True)
        transformer[:, 0] = 1 - transformer_m[:, 0]

        transformed = images * (1 - transformer_m) + 0.5 * transformer_m

        # for i in range(0, 20):
        #     d = transformer[0][i].clone().detach().cpu().numpy()
        #     cv2.imshow('f_' + str(i), d)

        # Show label for debugging
        cv2.imshow('transformer_lab', self.build_label(transformer[0]))

        # Show erase mask for debugging
        cv2.imshow('transformer_mas', transformer_m[0, 0].clone().detach().cpu().numpy())
        
        # Show erased image for debugging
        erased_input_vis = transformed[0].clone().detach().cpu().numpy()
        erased_input_vis = np.moveaxis(erased_input_vis, 0, -1)
        cv2.imshow('transformer_inp', erased_input_vis)
        cv2.waitKey(1)

        return transformed, transformer_c, transformer_m

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
        self.discriminator = Discriminator()
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

        interval_size = 100
        transformer_start = 300
        discriminator_train_start = 300

        # Adverserial training controller
        if self.training:
            if self.step_count // interval_size % 2 == 0 or self.step_count < transformer_start:
                with torch.no_grad():
                    transformation, transformer_pred, mask = self.transformer(image, classification_label)

                # Train Classifier
                classifier_input = None
                if random.random() < 0.4:
                    classifier_input = image
                else:
                    classifier_input = transformation

                classification = self.classifier(classifier_input)
                loss_bce_class = self.classifier.loss_bce(classification, classification_label)
                loss_bce_class.backward()
                self.classifier.optimizer.step()

                # Train Discriminator
                if self.step_count > discriminator_train_start:
                    discrimination_input = mask.clone()
                    discrimination_label = np.full((mask.shape[0], 1), 0.1)

                    for i in range(0, mask.shape[0]):
                        if random.random() < 0.5:
                            continue
                        else:
                            _, label_dict, _ = self.segmentation_loader.__getitem__(random.randint(0, self.segmentation_loader.__len__() -1))
                            seg = label_dict['segmentation']
                            seg = np.max(seg[1:], axis=0)
                            discrimination_input[i] = torch.tensor(seg, device=self.device, dtype=torch.float).unsqueeze(0)
                            discrimination_label[i] = 0.9

                    discrimination_input[discrimination_input > 0.9] = 0.9
                    discrimination_input[discrimination_input < 0.1] = 0.1

                    discrimination_label = torch.tensor(discrimination_label, device=self.device, dtype=torch.float)
                    
                    cv2.imshow('disc_mask', discrimination_input[0, 0].clone().detach().cpu().numpy())
                    cv2.waitKey(1)

                    discrimination = self.discriminator(discrimination_input)
                    loss_bce_disc = self.discriminator.loss_bce(discrimination, discrimination_label)
                    loss_bce_disc.backward()
                    self.discriminator.optimizer.step()
            else:
                # Train Transformer
                transformation, transformer_pred, mask = self.transformer(image, classification_label)
                classification = self.classifier(transformation)
                discrimination = self.discriminator(mask)

                discrimination_label = np.full((mask.shape[0], 1), 0.9)
                discrimination_label = torch.tensor(discrimination_label, device=self.device, dtype=torch.float)
                
                loss_classifier = torch.mean(classification[classification_label > 0.5])
                loss_transformer = self.transformer.loss_bce(transformer_pred, classification_label)
                loss_discriminator = self.discriminator.loss_bce(discrimination, discrimination_label)
                loss_mask = torch.mean(mask)

                attention_mining_loss = loss_classifier + loss_transformer + loss_mask + loss_discriminator
                attention_mining_loss.backward()
                self.transformer.optimizer.step()

            self.classifier.optimizer.zero_grad()
            self.transformer.optimizer.zero_grad()
            self.discriminator.optimizer.zero_grad()

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