def train_voc_unet_noskip():
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.unet_noskip import UNetNoSkip
    
    from data.voc2012_loader_selfsupervised import PascalVOCSelfsupervised

    # VGG16
    model = UNetNoSkip(outputs=3, name='voc_unet_noskip')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSelfsupervised('train'), batch_size=8, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSelfsupervised('val'), batch_size=8, shuffle=False, num_workers=6)
        },
        epochs=21,
    )