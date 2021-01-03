
if __name__ == '__main__':
    import sys, os, torch
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from torch.nn.modules.loss import MSELoss
    from torch.nn.modules.loss import BCELoss
    from training.train import train
    from metrics.accuracy import accuracy
    from metrics.f1 import f1
    from models.unet_adverserial import UNetAdverserial

    from data.voc2012_loader_classification import PascalVOCClassification

    # VGG16
    model = UNetAdverserial('unet_adverserial')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=4, shuffle=True, num_workers=4),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=4, shuffle=False, num_workers=4)
        },
        metrics={
            'classification': {
                'f1': f1,
            },
        },
        epochs=21,
        log_prefix='unet_adverserial'
    )