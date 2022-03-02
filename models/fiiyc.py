from artifacts.artifact_manager import artifact_manager
from data.loader_segmentation import VOCSegmentation
from torch.utils.data.dataloader import DataLoader
from data.voc2012 import label_to_image
import cv2, wandb, torch, random
import numpy as np
from models._common import ModelBase
from metrics.f1 import f1
from metrics.iou import iou
from training._common import move_to

from models._common import build_vgg_features, print_params

##################################################################################################################
# Classifier
##################################################################################################################
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            # VGGC(),
            build_vgg_features(),
            torch.nn.Conv2d(512, 20, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Sigmoid(),
        )
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        print_params(self.parameters(), "Classifier")

    def forward(self, image):
        return self.classifier(image)

##################################################################################################################
# Adversary
##################################################################################################################
class Adversary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.features = build_vgg_features()
        self.conv_comb = torch.nn.Conv2d(512, 21, 1)
        self.gmp = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.upsample = torch.nn.Upsample(scale_factor=16, mode='nearest')

        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        print_params(self.parameters(), "Adversary")

    def segment(self, image, c_label):
        c_label = c_label.clone()
        c_label[c_label > 0.5] = 1
        c_label[c_label < 0.5] = 0

        adversary = self.features(image)
        adversary = self.conv_comb(adversary)
        adversary = torch.sigmoid(adversary)

        segmentation = adversary.clone()
        segmentation = self.upsample(segmentation)
        segmentation[:, 0] = 0.3
        segmentation[:, 1:] *= c_label.unsqueeze(-1).unsqueeze(-1)

        classification = self.gmp(adversary[:, 1:])
        classification = torch.flatten(classification, 1, -1)

        return {
            'classification': classification,
            'segmentation': segmentation,
        }

    def forward(self, image, classification_label):
        result = self.segment(image, classification_label)

        # Generate erase mask
        segmentation = result["segmentation"]
        mask, _ = torch.max(segmentation[:, 1:], dim=1, keepdim=True)

        # Generate spot and erase images
        spot = image * mask
        erase = image * (1 - mask)

        result['spot'] = spot
        result['erase'] = erase
        result['mask'] = mask

        segmentation_np = segmentation.clone().detach().cpu().numpy()
        label_image_np = np.zeros((segmentation_np.shape[0], 256, 256, 3))

        for s in range(0, segmentation_np.shape[0]):
            label_image_np[s] = label_to_image(segmentation_np[s])

        result['vis_output'] = label_image_np
        result['vis_mask'] = np.moveaxis(mask.clone().detach().cpu().numpy(), 1, -1)
        result['vis_erase'] = np.moveaxis(erase.clone().detach().cpu().numpy(), 1, -1)
        result['vis_spot'] = np.moveaxis(spot.clone().detach().cpu().numpy(), 1, -1)

        return result

##################################################################################################################
# Super model
##################################################################################################################
class FIIYC(ModelBase):
    def __init__(self, **kwargs):
        super(FIIYC, self).__init__(**kwargs)
        self.step = 'adversary'
        self.step_count = 0
        self.step_count_a = 0
        self.step_count_c = 0
        self.step_count_vis = 0

        self.classifier = Classifier()
        self.adversary = Adversary()

        torch.backends.cudnn.deterministic = True
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)

        pair = next(iter(DataLoader(VOCSegmentation('val', dataset='voco'), batch_size=32, shuffle=True, num_workers=0)))
        self.demo_inputs = pair[0]
        self.dmeo_labels = pair[1]

    def event(self, event):
        super().event(event)

        if event['name'] == 'minibatch' and event['phase'] == 'train':
            image = event['inputs']['image']
            label_c = event['labels']['classification']
            label_s = event['labels']['segmentation']

            # Run input through adversary
            adversary_result = self.adversary(image, label_c)
            iou_label = label_s.clone().detach().cpu().numpy()
            iou_predi = adversary_result['segmentation'].clone().detach().cpu().numpy()
            max_indices = iou_predi.max(axis=1, keepdims=True) == iou_predi

            iou_predi = np.zeros(iou_predi.shape)
            iou_predi[max_indices] = 1
            score_iou = iou(iou_label[:, 1:], iou_predi[:, 1:])

            # Training controller
            self.step_count += 1
            if self.step == 'classifier':
                self.step_count_c += 1
                if self.step_count == 3:
                    self.step = 'adversary'
                    self.step_count = 0

                # Train classifier
                if random.random() > 0.5:
                    classification = self.classifier(image)
                else:
                    classification = self.classifier(adversary_result['erase'])
                loss_c_bce = self.classifier.loss_bce(classification, label_c)
                loss_c_bce.backward()
                self.classifier.optimizer.step()

                wandb.log({
                    "step_count_c": self.step_count_c,
                    "loss_c_bce": loss_c_bce,
                    "score_iou": score_iou
                })


            elif self.step == 'adversary':
                self.step_count_a += 1
                if self.step_count == 3:
                    self.step = 'classifier'
                    self.step_count = 0

                # Channel loss
                loss_a_channel = self.adversary.loss_bce(adversary_result['classification'], label_c)

                # Constrain loss
                loss_a_mask = torch.mean(adversary_result['mask'])

                # Classifier loss
                c_spot = self.classifier(adversary_result['spot'])
                c_erase = self.classifier(adversary_result['erase'])
                loss_a_spot = self.classifier.loss_bce(c_spot, label_c)
                loss_a_erase = torch.mean(c_erase[label_c > 0.5])

                # Get adversary final loss
                loss_a_final = loss_a_erase + loss_a_mask + loss_a_channel

                loss_a_final.backward()
                self.adversary.optimizer.step()

                wandb.log({
                    "step_count_a": self.step_count_a,
                    "loss_a_final": loss_a_final,
                    "loss_a_mask": loss_a_mask,
                    "loss_a_erase": loss_a_channel,
                    "loss_a_spot": loss_a_spot,
                    "loss_a_channel": loss_a_channel,
                    "score_iou": score_iou
                })

                # Visualize adversary progress
                if self.step_count_a % 10 == 0:
                    image = self.demo_inputs['image'].clone()
                    label = self.dmeo_labels['classification'].clone()
                    image = move_to(image, self.device)
                    label = move_to(label, self.device)

                    adversary_result = self.adversary(image, label)

                    for typez in ['vis_output', 'vis_mask', 'vis_erase', 'vis_spot']:
                        output = adversary_result[typez]
                        for i, o in enumerate(output):
                            cv2.imwrite(artifact_manager.getDir() + f'/{typez}_{i}_{self.step_count_vis}.png', o * 255)

                    self.step_count_vis += 1

            # Clear gradients
            self.classifier.optimizer.zero_grad()
            self.adversary.optimizer.zero_grad()

            cv2.imshow('image', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
            cv2.imshow('label', label_to_image(label_s[0].clone().detach().cpu().numpy()))
            cv2.imshow('output', adversary_result['vis_output'][0])
            cv2.imshow('mask', adversary_result['vis_mask'][0])
            cv2.imshow('erase', adversary_result['vis_erase'][0])
            cv2.imshow('spot', adversary_result['vis_spot'][0])

            cv2.waitKey(1)