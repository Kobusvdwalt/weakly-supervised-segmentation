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
        model = DeepLab()
        return model

    raise Error('Model name has no implementation')

def train_model(
    dataset_root,
    model_name = 'unet',
    epochs=51,
    batch_size=8,
    image_size=256,
):
    print('Model training : ', locals())
    from training.train import train
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation

    model = get_model(model_name)

    train(
        model=model,
        dataloaders = {
            'train': DataLoader(
                Segmentation(
                    dataset_root,
                    source='train',
                    image_size=image_size
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=6,
                pin_memory=True
            ),
            'val': DataLoader(
                Segmentation(
                    dataset_root,
                    source='val',
                    image_size=image_size
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=6,
                pin_memory=True,
            )
        },
        epochs=epochs,
        validation_mod=10
    )
