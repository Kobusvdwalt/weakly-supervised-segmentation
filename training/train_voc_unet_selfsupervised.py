
if __name__ == '__main__':
    import sys, os, torch
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from torch.nn.modules.loss import MSELoss
    from torch.nn.modules.loss import BCELoss
    from training.train import train
    from metrics.accuracy import accuracy
    from metrics.f1 import f1
    from models.unet_bottleneck import UNetBottleneck

    from data.voc2012_loader_selfsupervised import PascalVOCSelfsupervised
    #from data.voc2012_loader_classification import PascalVOCClassification

    # VGG16
    model = UNetBottleneck('unet_selfsupervised', 20)
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSelfsupervised('train'), batch_size=4, shuffle=True, num_workers=4),
            'val': DataLoader(PascalVOCSelfsupervised('val'), batch_size=4, shuffle=False, num_workers=4)
        },
        metrics={
            # 'classification': {
            #     'f1': f1,
            # },
            'reconstruction': {
                'accuracy': accuracy
            }
        },
        losses = {
            # 'classification': torch.nn.BCELoss(),
            'reconstruction': torch.nn.MSELoss()
        },
        epochs=21,
        log_prefix='unet_selfsupervised'
    )