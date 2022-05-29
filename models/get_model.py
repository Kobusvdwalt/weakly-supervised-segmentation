
from shutil import Error

def get_model(model_name):
    model = None
    if model_name == 'vgg16':
        from models.vgg16 import Vgg16GAP
        model = Vgg16GAP(name="vgg16")
        return model

    if model_name == 'unet':
        from models.unet import UNet
        model = UNet()
        return model

    if model_name == 'deeplab':
        from models.deeplab import DeepLab
        model = DeepLab(name="deeplab")
        return model

    if model_name == 'affinitynet':
        from models.aff_net import AffNet
        model = AffNet(name="affinitynet")
        return model

    if model_name == 'wasscam':
        from models.wass import WASS
        model = WASS()
        return model

    raise Error('Model name has no implementation')
