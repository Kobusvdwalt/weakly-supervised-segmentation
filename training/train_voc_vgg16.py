def train_voc_vgg16():
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.vgg16 import Vgg16GAP
    
    from data.voc2012_loader_classification import PascalVOCClassification

    # VGG16
    model = Vgg16GAP(name='voc_vgg16')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=16, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=16, shuffle=False, num_workers=6)
        },
        epochs=21,
    )