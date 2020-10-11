import torchvision
import torch
import os
import numpy as np
import cv2

class Vgg16GAP(torch.nn.Module):
    def __init__(self, name, outputs):
        super(Vgg16GAP, self).__init__()
        self.name = name + '_vgg16_gap_unfreeze_0'
        print(self.name)
        self.vgg = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg.features = self.vgg.features[:-1]
        self.vgg.avgpool = None
        self.vgg.classifier = None

        # Unfreeze last conv layer
        total = 0
        count = 0
        unfreeze = 0
        for param in self.vgg.parameters():
            total += 1
        for param in self.vgg.parameters():
            if (count >= total-unfreeze*2):
                print('layer unfrozen' + str(count))
                param.requires_grad = True
            else:
                param.requires_grad = False
            count += 1
        
        self.conv = torch.nn.Conv2d(512, outputs, 1)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # input_np = x.clone().detach().cpu().numpy()

        x = self.vgg.features(x)
        x = self.conv(x)
        
        # features_np = x.clone().detach().cpu().numpy()

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.sigmoid(x)

        # output_np = x.clone().detach().cpu().numpy()
        # output_np = np.moveaxis(output_np, 1, -1)
        # cv2.imshow('output', output_np[0])

        # Visualize Input
        # input_np = np.moveaxis(input_np, 1, -1)        
        # cv2.imshow('input', input_np[0])

        # # Visualize Features
        # count = features_np[0].shape[0]
        # cell_width = features_np[0].shape[1]
        # cell_height = features_np[0].shape[2]
        # instance = 0

        # stacked = features_np[0, 0]
        # stacked_h = None
        
        # scale = 16

        # cell_count_x = 7
        # cell_count_y = 3

        # grid = np.zeros((cell_count_y * cell_height * scale, cell_count_x * cell_width * scale))

        # for grid_y in range(0, cell_count_y):
        #     for grid_x in range(0, cell_count_x):
        #         if (instance >= count):
        #             break
        #         x_start = grid_x * cell_width * scale
        #         x_end = (grid_x+1) * cell_width * scale
        #         y_start = grid_y * cell_height * scale
        #         y_end = (grid_y+1) * cell_height * scale

        #         feat = cv2.resize(features_np[0, instance], (cell_width * scale, cell_height * scale), interpolation=cv2.INTER_NEAREST)
        #         feat[:, -1] = 1
        #         feat[:, 0] = 1
        #         feat[-1, :] = 1
        #         feat[0, :] = 1

        #         font                   = cv2.FONT_HERSHEY_PLAIN
        #         bottomLeftCornerOfText = (5, 25 + 5)
        #         fontScale              = 2
        #         fontColor              = (255,255,255)
        #         lineType               = 1

        #         cv2.putText(feat, str(instance),
        #             bottomLeftCornerOfText, 
        #             font, 
        #             fontScale,
        #             fontColor,
        #             lineType)

        #         grid[y_start: y_end, x_start: x_end] = feat
        #         instance += 1
        
        # cv2.imshow('features', grid)
        # cv2.waitKey(1)

        # Return output
        return x

    def load(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(package_directory, 'checkpoints', self.name + '.pt')
        self.load_state_dict(torch.load(weight_path))

    def save(self):
        print('saving model')
        package_directory = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(package_directory, 'checkpoints', self.name + '.pt')
        torch.save(self.state_dict(), weight_path)
