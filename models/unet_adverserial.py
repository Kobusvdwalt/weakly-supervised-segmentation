
from data.voc2012 import label_to_image
import cv2
import numpy as np
import torch, torchvision
import os
from models.model_base import ModelBase
from metrics.f1 import f1

def unfreeze_vgg_features(vgg):
    total = 0
    count = 0
    unfreeze = 2
    for param in vgg.parameters():
        total += 1
    for param in vgg.parameters():
        if (count >= total-unfreeze*2):
            param.requires_grad = True
        else:
            param.requires_grad = False
        count += 1

def freeze_vgg_features(vgg):
    for param in vgg.parameters():
        param.requires_grad = False

def build_vgg_features():
    vgg = torchvision.models.vgg16(pretrained=True, progress=True)
    vgg.avgpool = None
    vgg.classifier = None
    vgg.features = vgg.features[:-1]
    unfreeze_vgg_features(vgg.features)

    return vgg.features

##################################################################################################################
# Classifier
##################################################################################################################
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier_features = build_vgg_features()
        self.classifier_conv = torch.nn.Conv2d(512, 20, 1)
        self.classifier_gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier_sigmoid = torch.nn.Sigmoid()
        self.classifier_loss = torch.nn.BCELoss()
        self.classifier_optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def train(self, mode = True):
        super().train(mode)
        if mode == False:
            freeze_vgg_features(self.classifier_features)
            self.classifier_conv.weight.requires_grad = False
            self.classifier_conv.bias.requires_grad = False

        if mode == True:
            unfreeze_vgg_features(self.classifier_features)
            self.classifier_conv.weight.requires_grad = True
            self.classifier_conv.bias.requires_grad = True

    def apply_loss(self, outputs, labels):
        if self.training:
            loss = self.classifier_loss(outputs, labels)
            loss.backward(retain_graph=True)
            self.classifier_optimizer.step()
            self.classifier_optimizer.zero_grad()

    def forward(self, image):
        classification = self.classifier_features(image)
        classification = self.classifier_conv(classification)
        classification = self.classifier_gap(classification)
        classification = self.classifier_sigmoid(classification)
        classification = torch.flatten(classification, 1)

        return classification

##################################################################################################################
# Transformer
##################################################################################################################
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transformer_features = build_vgg_features()
        self.transformer_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest') # , align_corners=True
        self.transformer_conv1 = torch.nn.Conv2d(512, 256, 3, padding=1)

        self.transformer_conv2_1 = torch.nn.Conv2d(512 + 256, 128, 3, padding=1)
        self.transformer_conv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)

        self.transformer_conv3_1 = torch.nn.Conv2d(128 + 256, 64, 3, padding=1)
        self.transformer_conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)

        self.transformer_conv4_1 = torch.nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.transformer_conv4_2 = torch.nn.Conv2d(64, 64, 3, padding=1)

        self.transformer_conv5 = torch.nn.Conv2d(128, 20, 1)
        self.transformer_gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.transformer_gmp = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.transformer_relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer_sigmoid = torch.nn.Sigmoid()
        self.transformer_loss_func = torch.nn.BCELoss()
        self.transformer_optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.transformer_features[8].register_forward_hook(output_hook)
        self.transformer_features[15].register_forward_hook(output_hook)
        self.transformer_features[22].register_forward_hook(output_hook)

    def train(self, mode=True):
        super().train(mode)
        if mode == False:
            freeze_vgg_features(self.transformer_features)
            self.transformer_conv1.weight.requires_grad = False
            self.transformer_conv1.bias.requires_grad = False

            self.transformer_conv2_1.weight.requires_grad = False
            self.transformer_conv2_1.bias.requires_grad = False
            self.transformer_conv2_2.weight.requires_grad = False
            self.transformer_conv2_2.bias.requires_grad = False

            self.transformer_conv3_1.weight.requires_grad = False
            self.transformer_conv3_1.bias.requires_grad = False
            self.transformer_conv3_2.weight.requires_grad = False
            self.transformer_conv3_2.bias.requires_grad = False

            self.transformer_conv4_1.weight.requires_grad = False
            self.transformer_conv4_1.bias.requires_grad = False
            self.transformer_conv4_2.weight.requires_grad = False
            self.transformer_conv4_2.bias.requires_grad = False

            self.transformer_conv5.weight.requires_grad = False
            self.transformer_conv5.bias.requires_grad = False

            # print("\neval\n")
            
        if mode == True:
            unfreeze_vgg_features(self.transformer_features)
            self.transformer_conv1.weight.requires_grad = True
            self.transformer_conv1.bias.requires_grad = True

            self.transformer_conv2_1.weight.requires_grad = True
            self.transformer_conv2_1.bias.requires_grad = True
            self.transformer_conv2_2.weight.requires_grad = True
            self.transformer_conv2_2.bias.requires_grad = True

            self.transformer_conv3_1.weight.requires_grad = True
            self.transformer_conv3_1.bias.requires_grad = True
            self.transformer_conv3_2.weight.requires_grad = True
            self.transformer_conv3_2.bias.requires_grad = True

            self.transformer_conv4_1.weight.requires_grad = True
            self.transformer_conv4_1.bias.requires_grad = True
            self.transformer_conv4_2.weight.requires_grad = True
            self.transformer_conv4_2.bias.requires_grad = True

            self.transformer_conv5.weight.requires_grad = True
            self.transformer_conv5.bias.requires_grad = True

            # print("\ntrain\n")

    def apply_loss(self, classification, label):
        if self.training:
            # Attention Mining Loss. Mean of output of each class probability in image
            attention_mining_loss = torch.mean(classification[label>0.5])

            attention_mining_loss.backward(retain_graph=True)
            self.transformer_loss_bce.backward(retain_graph=True)
            # self.transformer_loss_reg.backward(retain_graph=True)
            self.transformer_optimizer.step()
            self.transformer_optimizer.zero_grad()
    
    def segment(self, image, label):
        self.intermediate_outputs.clear()
        transformer = self.transformer_features(image)
        transformer = self.transformer_conv1(transformer)
        transformer = self.transformer_relu(transformer)

        transformer = self.transformer_upsample(transformer)
        transformer = torch.cat((transformer, self.intermediate_outputs[2]), dim=1)
        transformer = self.transformer_conv2_1(transformer)
        transformer = self.transformer_relu(transformer)
        transformer = self.transformer_conv2_2(transformer)
        transformer = self.transformer_relu(transformer)
        
        # # 64
        # transformer = self.transformer_upsample(transformer)
        # transformer = torch.cat((transformer, self.intermediate_outputs[1]), dim=1)
        # transformer = self.transformer_conv3_1(transformer)
        # transformer = self.transformer_relu(transformer)
        # transformer = self.transformer_conv3_2(transformer)
        # transformer = self.transformer_relu(transformer)
        # # 128
        # transformer = self.transformer_upsample(transformer)
        # transformer = torch.cat((transformer, self.intermediate_outputs[0]), dim=1)
        # transformer = self.transformer_conv4_1(transformer)
        # transformer = self.transformer_relu(transformer)
        # transformer = self.transformer_conv4_2(transformer)
        # transformer = self.transformer_relu(transformer)

        transformer = self.transformer_conv5(transformer)
        transformer = self.transformer_sigmoid(transformer)

        dropout = torch.rand(transformer.shape, device = self.device)

        transformer_clean = transformer.clone()
        transformer_noise = transformer.clone() * dropout
        
        transformer_pred = self.transformer_gmp(transformer_noise)
        transformer_pred = torch.flatten(transformer_pred, 1)
        self.transformer_loss_bce = self.transformer_loss_func(transformer_pred, label)

        transformer_noise = self.transformer_upsample(transformer_noise)
        transformer_noise = self.transformer_upsample(transformer_noise)
        transformer_noise = self.transformer_upsample(transformer_noise)
        
        transformer_clean = self.transformer_upsample(transformer_clean)
        transformer_clean = self.transformer_upsample(transformer_clean)
        transformer_clean = self.transformer_upsample(transformer_clean)

        # 256
        # cv2.imshow('t_aeroplane', transformer_sig[0, 0].clone().detach().cpu().numpy())
        # cv2.imshow('t_bicycle', transformer_sig[0, 1].clone().detach().cpu().numpy())
        # cv2.imshow('t_bird', transformer_sig[0, 2].clone().detach().cpu().numpy())
        # cv2.imshow('t_boat', transformer_sig[0, 3].clone().detach().cpu().numpy())
        # cv2.imshow('t_person', transformer_sig[0, 14].clone().detach().cpu().numpy())
        # cv2.imshow('t_dog', transformer_sig[0, 7].clone().detach().cpu().numpy())
        return transformer_noise, transformer_clean

    def build_label(self, transformer, label):
        transformer_vis = transformer[0].clone().detach().cpu().numpy()
        label_vis = np.zeros((transformer_vis.shape[0]+1, transformer_vis.shape[1], transformer_vis.shape[2]))

        # Copy masks based on classification label
        for i in range(0, transformer_vis.shape[0]):
            if label[0, i] > 0.5:
                label_vis[i+1] = transformer_vis[i]


        # Compute background mask
        summed = np.mean(transformer_vis, 0)
        summed[summed > 1] = 1
        summed[summed < 0] = 0
        label_vis[0] = (1 - summed) * 0.5

        label_vis = label_to_image(label_vis)

        return label_vis


    def forward(self, images, labels):
        transformer_noise, transformer_clean = self.segment(images, labels)
        
        label_vis = self.build_label(transformer_clean, labels)
        cv2.imshow('label_vis', label_vis)

        transformed = images.clone()
        for batch_index in range(0, transformed.shape[0]):
            # Get activations from label
            activations = transformer_noise[batch_index, labels[batch_index] > 0.5]
            # Combine activations by max
            activation, _indices = torch.max(activations, 0)
            # Erase the image
            transformed[batch_index] = images[batch_index] * (1 - activation) * 0.5 + activation * 0.5
            
            # Show erase mask for debugging
            if batch_index == 0:
               activation_show = activation.clone().detach().cpu().numpy()
               cv2.imshow('t_comb', activation_show)
        
        # self.transformer_loss_reg = torch.mean(transformed) * 0.1

        # Show erased image for debugging
        erased_input_vis = transformed[0].clone().detach().cpu().numpy()
        erased_input_vis = np.moveaxis(erased_input_vis, 0, -1)
        cv2.imshow('erased_input_vis', erased_input_vis)
        cv2.waitKey(1)

        return transformed

class UNetAdverserial(ModelBase):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.step = -1
        self.step_count = 0

        self.classifier = Classifier()
        self.transformer = Transformer()

    def epoch_start(self):
        self.step = -1
        self.step_count = 0
        self.classifier.train(False)
        self.transformer.train(False)

    def forward(self, inputs):
        # Adverserial training controller
        if self.training:
            self.step_count += 1
            if self.step != (self.step_count // 50) % 2:
                self.step = (self.step_count // 50) % 2

                # Train classifier
                if self.step == 0:
                    self.classifier.train(True)
                    self.transformer.train(False)
                # Train transformer
                else:
                    self.classifier.train(False)
                    self.transformer.train(True)

        # Actual forward pass
        image = inputs['image']
        label = inputs['label']

        self.transformation = self.transformer(image, label)
        self.classification = self.classifier(self.transformation)

        outputs = {
            'classification': self.classification,
        }

        return outputs

    def backward(self, outputs, labels):
        if self.training:
            self.transformer.apply_loss(self.classification, labels['classification'])
            self.classifier.apply_loss(self.classification, labels['classification'])

    def segment(self, images, class_labels):
        x_noise, x_clean = self.transformer.segment(images, class_labels)

        # Build label
        result = np.zeros((images.shape[0], images.shape[2], images.shape[3], 3))

        for batch_index in range(0, images.shape[0]):
            features = x_clean[batch_index].clone().detach().cpu().numpy()
            label = np.zeros((features.shape[0]+1, features.shape[1], features.shape[2]))

            # Copy masks based on classification label
            for feature_index in range(0, features.shape[0]):
                if class_labels[batch_index, feature_index] > 0.5:
                    label[feature_index+1] = features[feature_index]


            # Compute background mask
            summed = np.mean(features, 0)
            summed[summed > 1] = 1
            summed[summed < 0] = 0
            label[0] = (1 - summed) * 0.5

            result[batch_index] = label_to_image(label)

        return result
            
    def metrics(self, outputs, labels):
        metrics = {
            'classification': {
                'f1': f1,
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
        return True