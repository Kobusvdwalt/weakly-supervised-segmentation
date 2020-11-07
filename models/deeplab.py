from data.voc2012 import label_to_image
import torchvision
import torch
import os, cv2
import numpy as np

# torch.utils.model_zoo.load_url( os.path.abspath('./checkpoints') )
# C:\Users\Kobus\.cache\torch\checkpoints
class DeepLab101(torch.nn.Module):
    def __init__(self, name, outputs, pretrained=True):
        super(DeepLab101, self).__init__()
        self.name = name
        self.deeplab101 = torchvision.models.segmentation.deeplabv3_resnet101 (pretrained=pretrained, progress=True)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        input_np = x.clone().detach().cpu().numpy()
        input_np = np.moveaxis(input_np, 1, -1)
        
        x = self.deeplab101(x)['out']
        x = self.sigmoid(x)

        output_np = x.clone().detach().cpu().numpy()
        output_np_image = label_to_image(output_np[0])
        cv2.imshow('input', input_np[0])
        cv2.imshow('output', output_np_image)
        cv2.waitKey(1)
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
