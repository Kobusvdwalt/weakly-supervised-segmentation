
import cv2, torch, random
import numpy as np

from torch.utils.data.dataloader import DataLoader
import wandb
from data.voc2012 import label_to_image, class_word_to_index
from models._common import ModelBase
from metrics.f1 import f1
from metrics.iou import iou, class_iou
from training._common import move_to
from tools.crf import CRF
from sklearn import metrics

from models._common import build_vgg_features, print_params

import torch.nn.functional as F

##################################################################################################################
# Classifier
##################################################################################################################
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            build_vgg_features(unfreeze_from=0),
            torch.nn.Conv2d(512, 20, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Sigmoid()
        )
        self.loss_bce = torch.nn.BCELoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.02, weight_decay=0.0001)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.000001)

    def forward(self, image):
        return self.classifier(image)

##################################################################################################################
# Adversary
##################################################################################################################
class Adversary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.features = build_vgg_features(unfreeze_from=0)
        self.lconv = self.double_conv(552, 512, 3, 1)
        self.dconv_up3 = self.double_conv(512, 256, 3, 1)
        self.dconv_up2 = self.double_conv(256, 128, 3, 1)
        self.dconv_up1 = self.double_conv(128, 64, 3, 1)

        self.conv_comb = torch.nn.Conv2d(64, 3, 1, bias=False)

        self.econv1 = self.double_conv(64, 16, 3, 1)
        # self.econv2 = self.double_conv(32, 16, 3, 1)
        self.econv_comb = torch.nn.Conv2d(16, 1, 1, bias=False)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        # self.features[3].register_forward_hook(output_hook)
        self.features[8].register_forward_hook(output_hook)
        self.features[15].register_forward_hook(output_hook)
        self.features[22].register_forward_hook(output_hook)

        self.pool2 = torch.nn.AvgPool2d((2, 2))
        self.pool4 = torch.nn.AvgPool2d((4, 4))
        self.pool8 = torch.nn.AvgPool2d((8, 8))
        self.gmp = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.loss_bce = torch.nn.BCELoss()
        self.loss_mse = torch.nn.MSELoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=0.0001)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.000001)

    def double_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
        )

    def segment(self, image, c_label, c_erase):
        # Build features
        features = self.features(image)

        # Integrate labels
        c_label = torch.unsqueeze(c_label, -1)
        c_label = torch.unsqueeze(c_label, -1)
        tensor_c_label = torch.ones((features.shape[0], c_label.shape[1], features.shape[2], features.shape[3]), device=self.device) * c_label
        
        c_erase = torch.unsqueeze(c_erase, -1)
        c_erase = torch.unsqueeze(c_erase, -1)
        tensor_c_erase = torch.ones((features.shape[0], c_erase.shape[1], features.shape[2], features.shape[3]), device=self.device) * c_erase

        features = torch.cat([features, tensor_c_label, tensor_c_erase], dim=1)
        features = self.lconv(features)

        # Upscale
        features = F.interpolate(features, scale_factor=2, mode='bilinear')
        features += self.intermediate_outputs[2]
        features = self.dconv_up3(features)

        features = F.interpolate(features, scale_factor=2, mode='bilinear')
        features += self.intermediate_outputs[1]
        features = self.dconv_up2(features)

        features = F.interpolate(features, scale_factor=2, mode='bilinear')
        features += self.intermediate_outputs[0]
        features = self.dconv_up1(features)

        # Reconstruct
        recon = self.conv_comb(features)

        # Erase
        erase = self.econv1(features)
        # erase = self.econv2(erase)
        erase = self.econv_comb(erase)

        self.intermediate_outputs.clear()
        
        return {
            'recon': recon,
            'erase': erase,
        }

def relut(x):
    return (1 / (1 + 10 * torch.pow(x, 2)))


##################################################################################################################
# Super model
##################################################################################################################
class WASS(ModelBase):
    def __init__(self, **kwargs):
        super(WASS, self).__init__(**kwargs)
        self.step = 0
        self.step_count = 0
        self.step_count_a = 0
        self.step_count_c = 0
        self.step_count_vis = 0

        self.classifier = Classifier()
        self.adversary = Adversary()

    def event(self, event):
        super().event(event)

        classifier_warmup = 50
        class_keep_probability = 0.5
        erase_image_probability = 0.5

        if event['name'] == 'minibatch':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_classification_cu = event['labels']['classification'].cuda(non_blocking=True)
            erase_classification_cu = label_classification_cu.clone()
            erase_classification_cu[torch.rand(erase_classification_cu.shape) > class_keep_probability] = 0

            # Run input through adversary
            adversary_result = self.adversary.segment(image_cu, label_classification_cu, erase_classification_cu)
            adversary_recon = adversary_result['recon']
            adversary_recon = torch.pow(adversary_recon, 2)

            image_cu_target = F.adaptive_avg_pool2d(image_cu, (adversary_recon.shape[2], adversary_recon.shape[3]))

            # Erase image
            adversary_erase = adversary_result['erase']
            adversary_erase = torch.sigmoid(adversary_erase)
            image_cu_erased = image_cu_target * (1 - adversary_erase) + 0.5 * adversary_erase

            # Train classifier
            if random.random() > erase_image_probability:
                c_classification = self.classifier(image_cu_target)
            else:
                c_classification = self.classifier(image_cu_erased)

            # Classifier loss
            c_loss = self.classifier.loss_bce(c_classification, label_classification_cu)

            # Reconstruction loss
            a_loss_reconstruction = F.mse_loss(adversary_recon, image_cu_target)

            # Constrain loss
            a_loss_constrain = torch.mean(adversary_erase) * 0.01

            # Classifier loss erased
            a_classification = self.classifier(image_cu_erased)
            # gt = label_classification_cu > 0.5
            # gt_c_erased = erase_classification_cu[gt]
            # gt_c_classi = a_classification[gt]
            a_loss_classifier_erase = F.mse_loss(a_classification, erase_classification_cu)

            # Final loss
            a_loss = a_loss_reconstruction + a_loss_classifier_erase + a_loss_constrain

            # Training controller
            self.step += 1
            if self.step % 2 == 0:
                self.step_count_c += 1
                
                self.classifier.optimizer.zero_grad()
                c_loss.backward()
                self.classifier.optimizer.step()

            elif self.step > classifier_warmup:
                self.step_count_a += 1

                self.adversary.optimizer.zero_grad()
                a_loss.backward()
                self.adversary.optimizer.step()

            wandb.log({
                'step': self.step,
                'c_loss': c_loss.detach().cpu().numpy(),
                'a_loss': a_loss.detach().cpu().numpy(),
                'a_loss_reconstruction': a_loss_reconstruction.detach().cpu().numpy(),
                'a_loss_classifier_erase': a_loss_classifier_erase.detach().cpu().numpy(),
            })

            if self.step % 16 == 0:
                mask_vis = adversary_erase[0, 0].detach().cpu().numpy()
                erased_vis = image_cu_erased[0].detach().cpu().numpy()
                target_vis = image_cu_target[0].detach().cpu().numpy()
                recon_vis = adversary_recon[0].detach().cpu().numpy()
                image_vis = image_cu[0].detach().cpu().numpy()

                erased_vis = np.moveaxis(erased_vis, 0, -1)
                recon_vis = np.moveaxis(recon_vis, 0, -1)
                image_vis = np.moveaxis(image_vis, 0, -1)
                target_vis = np.moveaxis(target_vis, 0, -1)

                mask_vis = cv2.resize(mask_vis, (image_vis.shape[0], image_vis.shape[1]), interpolation=cv2.INTER_NEAREST)
                target_vis = cv2.resize(target_vis, (image_vis.shape[0], image_vis.shape[1]), interpolation=cv2.INTER_NEAREST)
                recon_vis = cv2.resize(recon_vis, (image_vis.shape[0], image_vis.shape[1]), interpolation=cv2.INTER_NEAREST)
                erased_vis = cv2.resize(erased_vis, (image_vis.shape[0], image_vis.shape[1]), interpolation=cv2.INTER_NEAREST)

                cv2.imshow('mask_vis', mask_vis)
                cv2.imshow('erased_vis', erased_vis)
                cv2.imshow('recon_vis', recon_vis)
                cv2.imshow('image_vis', image_vis)
                cv2.imshow('target_vis', target_vis)
                cv2.waitKey(1)







                # label = label_classification_cu.detach().cpu().numpy()
                # predi = adversary_classification.detach().cpu().numpy().flatten()

                # predi[predi > 0.5] = 1
                # predi[predi <= 0.5] = 0

                # predi_vis = adversary_result[0].clone().detach().cpu().numpy()
                # predi_vis_bg = np.power(1 - np.max(predi_vis, axis=0, keepdims=True), 4)
                # predi_vis = np.concatenate((predi_vis_bg, predi_vis), axis=0)
                # cv2.imshow('predi', label_to_image(predi_vis))
























            #     # Segmentation loss
            #     segmentation_gen_label = adversary_result['segmentation'].clone().detach()
            #     segmentation_gen_label[:, 1:] *= torch.unsqueeze(torch.unsqueeze(classification_label, -1), -1)
            #     segmentation_gen_label[:, 0] = (1 - adversary_result['mask'][:, 0].detach())
            #     segmentation_gen_label_np = torch.softmax(segmentation_gen_label, dim=1).cpu().numpy()
            #     image_np = image.clone().detach().cpu().numpy()
            #     image_np = np.moveaxis(image_np, 1, -1)
            #     crf = CRF()
            #     result = crf.process(image_np[0], segmentation_gen_label_np[0])
            #     cv2.imshow('newlabel', label_to_image(result))
            #     loss_a_seg = self.adversary.loss_bce(adversary_result['segmentation'], segmentation_gen_label)

            #     # Constrain loss
            #     # loss_a_mask = torch.mean(adversary_result['segmentation']) * 0.01

            #     # Classifier loss
            #     # c_spot = self.classifier(adversary_result['spot'])
            #     # loss_a_spot = self.classifier.loss_bce(c_spot, classification_label)

            #     c_erase = self.classifier(adversary_result['erase'])
            #     loss_a_erase = torch.mean(c_erase[classification_label > 0.5]) * 0.1

            #     # Discrimination loss
            #     # discrimination = self.discriminator(adversary_result['mask'])
            #     # discrimination_label = torch.full((adversary_result['mask'].shape[0], 1), fill_value=0.9, device=self.device)
            #     # loss_a_disc = self.discriminator.loss_bce(discrimination, discrimination_label)

            #     # Get adversary final loss
            #     loss_a_final = loss_a_channel + loss_a_seg + loss_a_erase # + loss_a_mask # + loss_a_seg # + loss_a_disc loss_a_mask

            #     loss_a_final.backward()
            #     self.adversary.optimizer.step()

            #     wandb.log({
            #         "step_count_a": self.step_count_a,
            #         "loss_a_final": loss_a_final,
            #         # "loss_a_mask": loss_a_mask,
            #         # "loss_a_erase": loss_a_erase,
            #         # "loss_a_spot": loss_a_spot,
            #         "loss_a_channel": loss_a_channel,
            #         "score_iou": score_iou,
            #         "score_a_f1": f1(adversary_result['classification'].clone().detach().cpu().numpy(), classification_label.clone().detach().cpu().numpy()),
            #     })

            #     # Visualize adversary progress
            #     if self.step_count_a % 4 == 0:
            #         image = self.demo_inputs['image'].clone()
            #         label = self.dmeo_labels['classification'].clone()
            #         image = move_to(image, self.device)
            #         label = move_to(label, self.device)

            #         adversary_result = self.adversary(image, label)

            #         for typez in ['vis_output', 'vis_mask', 'vis_erase']:
            #             output = adversary_result[typez]
            #             for i, o in enumerate(output):
            #                 cv2.imwrite(artifact_manager.getDir() + f'/{typez}_{i}_{self.step_count_vis}.png', o * 255)

            #         self.step_count_vis += 1

            # self.discriminator.optimizer.zero_grad()

            # for i in range(adversary_result['segmentation'].shape[1]):
            #     cv2.imshow(str(i), adversary_result['segmentation'][0][i].clone().detach().cpu().numpy())

            # cv2.imshow('image', np.moveaxis(image[0].clone().detach().cpu().numpy(), 0, -1))
            # cv2.imshow('label', label_to_image(segmentation_label[0].clone().detach().cpu().numpy()))
            # cv2.imshow('output', adversary_result['vis_output'][0])
            # cv2.imshow('mask', adversary_result['vis_mask'][0])
            # cv2.imshow('erase', adversary_result['vis_erase'][0])

            # cv2.waitKey(1)






        # self.discriminator = Discriminator()
        # self.segmentation_loader = VOCSegmentation('train', dataset='voco')
        # pair = next(iter(DataLoader(VOCSegmentation('val', dataset='voco'), batch_size=32, shuffle=True, num_workers=0)))
        # self.demo_inputs = pair[0]
        # self.dmeo_labels = pair[1]









# class Discriminator(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.discriminator = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 16, kernel_size=4, stride=2),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
#             torch.nn.LeakyReLU(negative_slope=0.11),
#             torch.nn.Conv2d(128, 1, 1, padding=0),
#             torch.nn.Sigmoid(),
#             torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#             torch.nn.Flatten(1, 3),
#         )
#         self.loss_bce = torch.nn.BCELoss()
#         # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.02, momentum=0.7)
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)

#     def forward(self, image):
#         return self.discriminator(image)





















# def gaus_kernel(shape=(3,3),sigma=10):
#     """
#     2D gaussian mask - should give the same result as MATLAB's
#     fspecial('gaussian',[shape],[sigma])
#     """
#     m,n = [(ss-1.)/2. for ss in shape]
#     y,x = np.ogrid[-m:m+1,-n:n+1]
#     h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
#     h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
#     sumh = h.sum()
#     if sumh != 0:
#         h /= sumh
#     return h

# class Blobber(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         kernel_size = 27

#         kernel = gaus_kernel(shape=(kernel_size, kernel_size))
#         conv = torch.nn.Conv2d(1, 1, kernel_size, padding=(kernel_size-1)//2)
#         conv.bias.data.fill_(0)
#         conv.weight.data.copy_(torch.tensor(kernel))

#         self.blob_conv = conv

#     def blur(self, input):
#         s = input.clone()
#         for b in range(input.shape[0]):
#             for c in range(input.shape[1]):
#                 s[b, c] = self.blob_conv(input[b:b+1, c:c+1])[0]

#         return s

#     def forward(self, input):
#         s = self.blur(input)
#         return s































                # wandb.log({
                #     "step_count_c": self.step_count_c,
                #     "loss_c_bce": loss_c_bce,
                #     "score_iou": score_iou,
                #     "score_c_f1": f1(classification.clone().detach().cpu().numpy(), classification_label.clone().detach().cpu().numpy()),
                #     # "loss_d_bce": loss_d_bce,
                #     # "score_d_f1": f1(discrimination.clone().detach().cpu().numpy(), discrimination_label.clone().detach().cpu().numpy())
                # })










                # Train discriminator
                # discrimination_input = adversary_result['mask'].clone()
                # discrimination_label = np.full((adversary_result['mask'].shape[0], 1), 0.1)

                # for i in range(0, adversary_result['mask'].shape[0]):
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


    # def forward(self, image, classification_label):
    #     result = self.segment(image, classification_label)

    #     # Generate erase mask
    #     segmentation = result["segmentation"]
    #     mask, _ = torch.max(segmentation[:, 1:], dim=1, keepdim=True)

    #     # Generate spot and erase images
    #     spot = image * mask
    #     erase = image * (1 - mask)
        
    #     result['spot'] = spot
    #     result['erase'] = erase
    #     result['mask'] = mask

    #     segmentation_np = segmentation.clone().detach().cpu().numpy()
    #     label_image_np = np.zeros((segmentation_np.shape[0], 256, 256, 3))

    #     for s in range(0, segmentation_np.shape[0]):
    #         label_image_np[s] = label_to_image(segmentation_np[s])
        
    #     result['vis_output'] = label_image_np
    #     result['vis_mask'] = np.moveaxis(mask.clone().detach().cpu().numpy(), 1, -1)
    #     result['vis_erase'] = np.moveaxis(erase.clone().detach().cpu().numpy(), 1, -1)
    #     result['vis_spot'] = np.moveaxis(spot.clone().detach().cpu().numpy(), 1, -1)
        
    #     return result