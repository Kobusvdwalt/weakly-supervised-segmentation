if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from metrics.accuracy import accuracy
    from data import voc2012
    from models.unet import UNet

    from data.voc2012_loader_selfsupervised import PascalVOCSelfsupervised

    # VGG16
    model = UNet('unet_selfsupervised', 3)
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSelfsupervised('train'), batch_size=4, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSelfsupervised('val'), batch_size=4, shuffle=False, num_workers=6)
        },
        metrics={
            'accuracy': accuracy,
        },
        epochs=21,
        log_prefix='unet_selfsupervised'
    )

