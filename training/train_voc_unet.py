
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.unet import UNet
    
    from data.voc2012_loader_segmentation import PascalVOCSegmentation

    # VGG16
    model = UNet(outputs=21, name='voc_unet')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentation('train'), batch_size=4, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=4, shuffle=False, num_workers=6)
        },
        epochs=21,
        log_prefix='unet'
    )

