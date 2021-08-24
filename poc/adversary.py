import torchvision
import torch

class Adversary(torch.nn.Module):
    def __init__(self, class_count=20):
        super().__init__()
        
        # Get a VGG16 network, pretrained on imagenet
        vgg = torchvision.models.vgg16(pretrained=True, progress=True)

        # Drop the linear layers
        self.features = vgg.features

        # Drop the last max pooling layer
        self.features = self.features[:-1]

        # Unfreeze last couple of convolutional layers
        for param_count, param in enumerate(self.features.parameters()):
            # There are two sets of params, weights and biases.
            # So we unfreeze 2 params per layer
            if param_count <= 2 * 6:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Build up convolutions
        self.dconv_up3 = self.double_conv(512, 256, 3, 1)
        self.dconv_up2 = self.double_conv(256, 128, 3, 1)
        self.dconv_up1 = self.double_conv(128, 64, 3, 1)
        self.dconv_up0 = self.double_conv(64, 64, 3, 1)
        self.conv_comb = torch.nn.Conv2d(64, class_count, 3, padding=1)

        # Set up hooks for skip connections
        self.intermediate_outputs = []
        def output_hook(module, input, output):
            self.intermediate_outputs.append(output)

        self.features[3].register_forward_hook(output_hook)
        self.features[8].register_forward_hook(output_hook)
        self.features[15].register_forward_hook(output_hook)
        self.features[22].register_forward_hook(output_hook)

        # Other utilities
        self.sigmoid = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten(1, -1)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Loss and optimizer
        self.loss_bce = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)

    def double_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.LeakyReLU(negative_slope=0.11, inplace=True),
        )

    def training_pass(self, image, classification_label):
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

        # Select true classes
        segmentation = adversary.clone()
        segmentation[:, 0] = 0.51
        segmentation[:, 1:] *= classification_label.unsqueeze(-1).unsqueeze(-1)
        segmentation = torch.sigmoid((segmentation -0.5) * 100)

        # Generate erase mask
        mask, _ = torch.max(segmentation[:, 1:], dim=1, keepdim=True)
        erase = image * (1 - mask)

        return {
            'erase_mask': mask,
            'erase_image': erase,
            'classification': classification
        }