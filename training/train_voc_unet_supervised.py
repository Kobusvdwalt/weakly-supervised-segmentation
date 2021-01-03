
if __name__ == '__main__':
    import sys, os, torch
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from metrics.iou import iou
    from data import voc2012
    from models.unet import UNet

    from data.voc2012_loader_segmentation import PascalVOCSegmentation

    # VGG16
    model = UNet('voc_segmentation_supervised', voc2012.get_class_count())
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentation('train'), batch_size=4, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=4, shuffle=False, num_workers=6)
        },
        metrics={
            'segmentation': {
                'miou': iou,
            }
        },
        losses = {
            'segmentation': torch.nn.BCELoss()
        },
        epochs=21,
        log_prefix='unet_supervised'
    )

