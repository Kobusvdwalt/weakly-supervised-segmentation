
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.vgg16_gap_feat import Vgg16GAP
    
    from data.voc2012_loader_classification import PascalVOCClassification

    # VGG16
    model = Vgg16GAP(name='voc_classification')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=16, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=16, shuffle=False, num_workers=6)
        },
        epochs=21,
        log_prefix='vgg_16'
    )

