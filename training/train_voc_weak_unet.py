def train_voc_weak_unet(vis_folder):
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.unet import UNet
    
    from data.voc2012_loader_weak import PascalVOCSegmentationWeak

    model = UNet(outputs=21, name='voc_unet_weak')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentationWeak('train', vis_folder=vis_folder), batch_size=8, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentationWeak('val', vis_folder=vis_folder), batch_size=8, shuffle=False, num_workers=6)
        },
        epochs=21,
    )