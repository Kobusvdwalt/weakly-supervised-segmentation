def train_voc_unet():
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.unet import UNet
    from training._common import Checkpointer, Visualizer
    from data.voc2012_loader_segmentation import PascalVOCSegmentation

    model = UNet(outputs=21, name='voc_unet', event_consumers=[Visualizer()])
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentation('train'), batch_size=32, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
        },
        epochs=21,
    )