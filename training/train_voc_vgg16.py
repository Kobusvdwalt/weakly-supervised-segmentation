
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from metrics.f1 import f1
    from data import voc2012
    import torch
    from models.vgg16_gap_feat import Vgg16GAP
    
    from data.voc2012_loader_classification import PascalVOCClassification

    # VGG16
    model = Vgg16GAP('voc_classification', voc2012.get_class_count() -1)
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=16, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=16, shuffle=False, num_workers=6)
        },
        metrics={
            'classification': {
                'f1': f1,
            }
        },
        losses = {
            'classification': torch.nn.BCELoss()
        },
        epochs=21,
        log_prefix='vgg_16'
    )

