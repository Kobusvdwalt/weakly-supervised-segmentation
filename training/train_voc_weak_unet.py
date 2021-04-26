def train_voc_weak_unet(vis_folder):
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.unet import UNet
    from training.helpers import Checkpointer, Visualizer
    
    from data.voc2012_loader_weak import PascalVOCSegmentationWeak

    model = UNet(outputs=21, name='voc_unet_weak', event_consumers=[Visualizer()])
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentationWeak('train', vis_folder=vis_folder), batch_size=32, shuffle=True, num_workers=8),
            'val': DataLoader(PascalVOCSegmentationWeak('val', vis_folder=vis_folder), batch_size=32, shuffle=False, num_workers=8)
        },
        epochs=21,
    )