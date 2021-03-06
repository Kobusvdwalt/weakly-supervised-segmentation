if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from metrics.iou import iou
    from data import voc2012
    from models.unet import UNet

    from data.voc2012_loader_weak import PascalVOCSegmentationWeak
    from data.voc2012_loader_segmentation import PascalVOCSegmentation

    # VGG16
    model = UNet('voc_segmentation_weak', voc2012.get_class_count())
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentationWeak('train', '../visualize/output/voc_classification_vgg16_gap_val/'), batch_size=4, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=4, shuffle=False, num_workers=6)
        },
        metrics={
            'miou': iou,
        },
        epochs=21,
        log_prefix='unet_101_weak'
    )

