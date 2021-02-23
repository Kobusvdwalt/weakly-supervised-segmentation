def train_voc_deeplab101():
    from models.deeplab101 import DeepLab101
    from training.train import train
    from torch.utils.data.dataloader import DataLoader
    from data.voc2012_loader_segmentation import PascalVOCSegmentation

    model = DeepLab101(outputs=21, name='voc_deeplab101')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentation('train'), batch_size=2, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=2, shuffle=False, num_workers=6)
        },
        epochs=21,
    )