from artifacts.artifact_manager import artifact_manager
from data.loader_segmentation import VOCSegmentation
from torch.utils.data.dataloader import DataLoader
from data.voc2012 import label_to_image
import cv2, wandb, torch, random
import numpy as np
from models._common import ModelBase
from metrics.f1 import f1
from training._common import move_to

from models._common import build_vgg_features, print_params

##################################################################################################################
# Mask Discriminator
##################################################################################################################
class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.11),
            torch.nn.Conv2d(128, 1, 1, padding=0),
            torch.nn.Sigmoid(),
            torch.nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
        )
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, image):
        return self.discriminator(image)

##################################################################################################################
# Classifier
##################################################################################################################
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            build_vgg_features(),
            torch.nn.Conv2d(512, 20, 1, padding=0),
            torch.nn.Sigmoid(),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
        )
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
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
        self.dconv_up3 = self.double_conv(512, 256, 3, 1)
        self.dconv_up2 = self.double_conv(256, 128, 3, 1)
        self.dconv_up1 = self.double_conv(128, 64, 3, 1)
        self.dconv_up0 = self.double_conv(64, 64, 3, 1)
        self.conv_comb = torch.nn.Conv2d(64, 21, 3, padding=1)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.features[3].register_forward_hook(output_hook)
        self.features[8].register_forward_hook(output_hook)
        self.features[15].register_forward_hook(output_hook)
        self.features[22].register_forward_hook(output_hook)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(1)
        self.flatten = torch.nn.Flatten(1, -1)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.dnz = torch.nn.AvgPool2d(kernel_size=8)
        self.upz = torch.nn.Upsample(scale_factor=8, mode='nearest')
        self.upz_16 = torch.nn.Upsample(scale_factor=16, mode='nearest')
        self.upz_8 = torch.nn.Upsample(scale_factor=8, mode='nearest')
        
        
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.loss_bce = torch.nn.BCELoss()
        self.loss_cce = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
        print_params(self.parameters(), "Adversary")

    def double_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
        )

    def segment(self, image):
        adversary = self.features(image)

        adversary = self.upsample(adversary)
        adversary += self.intermediate_outputs[3]
        adversary = self.dconv_up3(adversary)

        adversary = self.upsample(adversary)
        adversary += self.intermediate_outputs[2]
        adversary = self.dconv_up2(adversary)

        adversary = self.upsample(adversary)
        adversary += self.intermediate_outputs[1]
        adversary = self.dconv_up1(adversary)

        adversary = self.upsample(adversary)
        adversary += self.intermediate_outputs[0]
        adversary = self.dconv_up0(adversary)

        adversary = self.conv_comb(adversary)
        adversary = self.sigmoid(adversary)

        classification = self.gap(adversary[:, 1:])
        classification = self.flatten(classification)

        self.intermediate_outputs.clear()

        return {
            'adversary': adversary,
            'classification': classification
        }

    def forward(self, image, classification_label):
        result = self.segment(image)

        rand_32 = torch.rand((result["adversary"].shape[0], 1, 32, 32), device=self.device)
        rand_32[rand_32 >= 0.2] = 1
        rand_32[rand_32 < 0.2] = 0
        rand_32 = self.upz_8(rand_32)

        # Select true classes
        segmentation = result["adversary"].clone()
        # segmentation[:, 0] = 0.51
        # segmentation[:, 1:] *= classification_label.unsqueeze(-1).unsqueeze(-1)
        segmentation = torch.sigmoid((segmentation -0.5) * 100)

        # Generate erase mask
        # mask, _ = torch.max(segmentation[:, 1:], dim=1, keepdim=True)
        mask = segmentation[:, 1:2, :, :]
        erase = image * mask

        mask_ds = self.upz(self.dnz(mask))
        mask_ds = mask_ds * rand_32
        erase_ds = image * (1-mask_ds)

        result['erase_ds'] = erase_ds
        result['erase_og'] = erase
        result['mask_og'] = mask

        segmentation_np = segmentation.clone().detach().cpu().numpy()
        label_image_np = np.zeros((segmentation_np.shape[0], 256, 256, 3))

        for s in range(0, segmentation_np.shape[0]):
            label_image_np[s] = label_to_image(segmentation_np[s])
        
        result['vis_output'] = label_image_np
        result['vis_mask_og'] = np.moveaxis(mask.clone().detach().cpu().numpy(), 1, -1)
        result['vis_erase_og'] = np.moveaxis(erase.clone().detach().cpu().numpy(), 1, -1)
        return result

##################################################################################################################
# Super model
##################################################################################################################
class WASS(ModelBase):
    def __init__(self, **kwargs):
        super(WASS, self).__init__(**kwargs)
        self.step = 'adversary'
        self.step_count = 0
        self.step_count_vis = 0

        self.classifier = Classifier()
        self.adversary = Adversary()
        self.discriminator = Discriminator()
        self.segmentation_loader = VOCSegmentation('train', dataset='voco')

        torch.backends.cudnn.deterministic = True
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)

        pair = next(iter(DataLoader(VOCSegmentation('val', dataset='voco'), batch_size=32, shuffle=True, num_workers=0)))
        self.demo_inputs = pair[0]
        self.dmeo_labels = pair[1]

    def new_instance(self):
        return WASS(name=self.name)

    def event(self, event):
        super().event(event)

        if event['name'] == 'minibatch' and event['phase'] == 'train':
            loss_c_bce = 0
            loss_a_bce = 0
            loss_a_mask = 0
            loss_a_mining = 0
            loss_a_final = 0
            loss_d_bce = 0

            image = event['inputs']['image']
            label_c = event['labels']['classification']
            label_s = event['labels']['segmentation']

            # Training controller
            self.step_count += 1
            if self.step == 'classifier':
                if self.step_count == 4:
                    self.step = 'adversary'
                    self.step_count = 0

                # Run input through adversary
                with torch.no_grad():
                    adversary_result = self.adversary(image, label_c)

                # Train classifier
                if random.random() < 0.8:
                    classification = self.classifier(image)
                else:
                    classification = self.classifier(adversary_result['erase_og'])

                loss_c_bce = self.classifier.loss_bce(classification, label_c)
                loss_c_bce.backward()
                self.classifier.optimizer.step()

                # Train discriminator
                # discrimination_input = adversary_result['mask_og'].clone()
                # discrimination_label = np.full((adversary_result['mask_og'].shape[0], 1), 0.1)

                # for i in range(0, adversary_result['mask_og'].shape[0]):
                #     if random.random() < 0.5:
                #         continue
                #     else:
                #         _, label_dict, _ = self.segmentation_loader.__getitem__(random.randint(0, self.segmentation_loader.__len__() -1))
                #         seg = label_dict['segmentation']
                #         seg = np.max(seg[1:], axis=0)
                #         discrimination_input[i] = torch.tensor(seg, device=self.device, dtype=torch.float).unsqueeze(0)
                #         discrimination_label[i] = 0.9

                # discrimination_label = torch.tensor(discrimination_label, device=self.device, dtype=torch.float)
                
                # cv2.imshow('disc_mask', discrimination_input[0, 0].clone().detach().cpu().numpy())
                # cv2.waitKey(1)

                # discrimination = self.discriminator(discrimination_input)
                # loss_d_bce = self.discriminator.loss_bce(discrimination, discrimination_label)
                # loss_d_bce.backward()
                # self.discriminator.optimizer.step()

            elif self.step == 'adversary':
                if self.step_count == 4:
                    self.step = 'classifier'
                    self.step_count = 0

                adversary_result = self.adversary(image, label_c)

                # Get adversary classification loss
                # loss_a_bce_c = self.adversary.loss_bce(adversary_result['classification'], label_c)

                # Get adversary mask loss
                loss_a_mask = torch.mean(adversary_result['mask_og']) * 0.1
                
                # Get adversary mining loss
                classification = self.classifier(adversary_result['erase_og'])
                loss_a_mining = self.classifier.loss_bce(classification, label_c) #  torch.mean(classification[label_c > 0.5])

                # Get adversary segmentation loss
                # loss_a_bce_s = self.adversary.loss_bce(adversary_result['adversary'], label_s)

                # Get adversary discrimination loss
                # discrimination = self.discriminator(adversary_result['mask_og'])
                # discrimination_label = torch.full((adversary_result['mask_og'].shape[0], 1), fill_value=0.9, device=self.device)
                # loss_a_disc = self.discriminator.loss_bce(discrimination, discrimination_label)

                # Get adversary final loss
                loss_a_final = loss_a_mining + loss_a_mask # loss_a_bce_s # + loss_a_disc

                loss_a_final.backward()
                self.adversary.optimizer.step()

                # Visualize adversary progress
                if self.step_count % 10 == 0:
                    image = self.demo_inputs['image'].clone()
                    label = self.dmeo_labels['classification'].clone()
                    image = move_to(image, self.device)
                    label = move_to(label, self.device)

                    adversary_result = self.adversary(image, label)

                    for typez in ['vis_output', 'vis_mask_og']:
                        output = adversary_result[typez]
                        for i, o in enumerate(output):
                            cv2.imwrite(artifact_manager.getDir() + f'/{typez}_{i}_{self.step_count_vis}.png', o * 255)

                    self.step_count_vis += 1

            # Clear gradients
            self.classifier.optimizer.zero_grad()
            self.adversary.optimizer.zero_grad()
            self.discriminator.optimizer.zero_grad()

            # Log
            # wandb.log({
            #     "loss_c_bce": loss_c_bce,
            #     "loss_t_bce": loss_a_bce,
            #     "loss_t_mask": loss_a_mask,
            #     "loss_t_mining": loss_a_mining,
            #     "loss_t_final": loss_a_final,
            #     "loss_d_bce": loss_d_bce
            # })

            cv2.imshow('image', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
            cv2.imshow('label', label_to_image(label_s[0].clone().detach().cpu().numpy()))
            cv2.imshow('output', adversary_result['vis_output'][0])
            cv2.imshow('mask_og', adversary_result['vis_mask_og'][0])
            cv2.imshow('erase_og', adversary_result['vis_erase_og'][0])

            cv2.waitKey(1)