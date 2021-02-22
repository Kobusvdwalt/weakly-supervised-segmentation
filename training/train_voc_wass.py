
if __name__ == '__main__':
    import sys, os, torch
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.wass import WASS

    from data.voc2012_loader_classification import PascalVOCClassification

    # VGG16
    model = WASS('voc_wass')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=4, shuffle=True, num_workers=4),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=4, shuffle=False, num_workers=4)
        },
        epochs=21,
        log_prefix='voc_wass'
    )