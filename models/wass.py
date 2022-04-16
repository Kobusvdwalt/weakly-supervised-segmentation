
import cv2, torch, random
import numpy as np

from torch.utils.data.dataloader import DataLoader
from data.voc2012 import label_to_image, class_word_to_index
from models._common import ModelBase
from metrics.f1 import f1
from metrics.iou import iou, class_iou
from training._common import move_to
from tools.crf import CRF

from models._common import build_vgg_features, print_params

import torch.nn.functional as F

##################################################################################################################
# Classifier
##################################################################################################################
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            build_vgg_features(),
            torch.nn.Conv2d(512, 20, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(1, 3),
            torch.nn.Sigmoid()
        )
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.02, momentum=0.7)
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
        self.dconv_up0 = self.double_conv(64, 32, 3, 1)
        self.conv_comb = torch.nn.Conv2d(128, 20, 1)

        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.features[3].register_forward_hook(output_hook)
        self.features[8].register_forward_hook(output_hook)
        self.features[15].register_forward_hook(output_hook)
        self.features[22].register_forward_hook(output_hook)

        self.upsample_low = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.gmp = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.7)
        print_params(self.parameters(), "Adversary")

    def double_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
        )

    def segment(self, image, c_label):
        c_label = c_label.clone()
        c_label[c_label > 0.5] = 1
        c_label[c_label < 0.5] = 0

        adversary = self.features(image)

        adversary = self.upsample(adversary)
        adversary += self.intermediate_outputs[3]
        adversary = self.dconv_up3(adversary)

        adversary = self.upsample(adversary)
        adversary += self.intermediate_outputs[2]
        adversary = self.dconv_up2(adversary)

        # adversary = self.upsample(adversary)
        # adversary += self.intermediate_outputs[1]
        # adversary = self.dconv_up1(adversary)

        # adversary = self.upsample(adversary)
        # adversary += self.intermediate_outputs[0]
        # adversary = self.dconv_up0(adversary)

        # adversary = self.upsample(adversary)
        adversary = self.upsample(adversary)
        adversary = self.upsample(adversary)

        adversary = self.conv_comb(adversary)

        self.intermediate_outputs.clear()
        
        return adversary



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


        torch.backends.cudnn.deterministic = True
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)

    def event(self, event):
        super().event(event)

        # if event['name'] == 'get_cam':
        #     image_cu = event['inputs']['image'].cuda(non_blocking=True)
        #     result_cu = self.segment(image_cu)
        #     return result_cu.detach().cpu().numpy()

        if event['name'] == 'minibatch':
            image_cu = event['inputs']['image'].cuda(non_blocking=True)
            label_classification_cu = event['labels']['classification'].cuda(non_blocking=True)

            # Run input through adversary
            adversary_result = self.adversary.segment(image_cu, label_classification_cu)
            # a_min, _ = torch.min(adversary_result, dim=0, keepdim=True)
            # a_max, _ = torch.max(adversary_result, dim=0, keepdim=True)
            # adversary_result = (adversary_result - a_min) / (a_max + 0.0001)
            # v_min, v_max = v.min(), v.max()

            adversary_result = torch.sigmoid(adversary_result) # TODO: try a linear normalization
            mask, _ = torch.max(adversary_result, dim=1, keepdim=True)
            erased = image_cu * (1 - mask)

            # Training controller
            self.step += 1
            if self.step % 2 == 0:
                self.step_count_c += 1

                # Train classifier
                if random.random() > 0.5:
                    classification = self.classifier(image_cu)
                else:
                    classification = self.classifier(erased)

                loss_c_bce = self.classifier.loss_bce(classification, label_classification_cu)
                self.classifier.optimizer.zero_grad()
                loss_c_bce.backward()
                self.classifier.optimizer.step()

            else:
                self.step_count_a += 1
                
                # Train adversary
                # Channel loss
                adversary_classification = F.adaptive_avg_pool2d(adversary_result, [1, 1])
                adversary_classification = torch.flatten(adversary_classification, 1)
                loss_a_channel = self.adversary.loss_bce(adversary_classification[label_classification_cu > 0.5], label_classification_cu[label_classification_cu > 0.5])

                # Classifier loss
                mask, _ = torch.max(adversary_result, dim=1, keepdim=True)
                erased = image_cu * (1 - mask)
                erased[erased <= 0] = 0.00001
                classification = self.classifier(erased)
                loss_a_classifier = torch.mean(classification[label_classification_cu > 0.5])

                # Constrain loss
                loss_a_constrain = torch.mean(adversary_result[adversary_result > 0.5]) + 0.0001
                loss_a = loss_a_classifier * 0.5 + loss_a_constrain * 0.3 + loss_a_channel * 0.2
                print(loss_a)

                self.adversary.optimizer.zero_grad()
                loss_a.backward()
                self.adversary.optimizer.step()

            erased = erased[0].detach().cpu().numpy()
            erased = np.moveaxis(erased, 0, -1)
            mask_vis = mask[0, 0].detach().cpu().numpy()
            mask_min = np.min(mask_vis)
            mask_normed = mask_vis - mask_min
            mask_max = np.max(mask_normed)
            mask_normed = mask_normed / (mask_max + 1e-5)
            cv2.imshow('erased', erased)
            cv2.imshow('mask_vis', mask_vis)
            cv2.imshow('mask_normed', mask_normed)
            cv2.waitKey(1)






























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