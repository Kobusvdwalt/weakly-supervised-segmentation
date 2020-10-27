if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from metrics.f1 import f1
    from data import voc2012
    from models.vgg16_gap_feat import Vgg16GAP
    
    from data.voc2012_loaders import PascalVOCClassification

    # VGG16
    # model = Vgg16GAP('voc_classification_epoch_1', voc2012.get_class_count() -1)
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(PascalVOCClassification('train'), batch_size=16, shuffle=True, num_workers=6),
    #         'val': DataLoader(PascalVOCClassification('val'), batch_size=16, shuffle=False, num_workers=6)
    #     },
    #     metrics={
    #         'f1': f1,
    #     },
    #     epochs=1,
    #     log_prefix='vgg_16_epoch_1'
    # )

    # model = Vgg16GAP('voc_classification_epoch_5', voc2012.get_class_count() -1)
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(PascalVOCClassification('train'), batch_size=16, shuffle=True, num_workers=6),
    #         'val': DataLoader(PascalVOCClassification('val'), batch_size=16, shuffle=False, num_workers=6)
    #     },
    #     metrics={
    #         'f1': f1,
    #     },
    #     epochs=6,
    #     log_prefix='vgg_16_epoch_5'
    # )

    model = Vgg16GAP('voc_classification_epoch_10', voc2012.get_class_count() -1)
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=16, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=16, shuffle=False, num_workers=6)
        },
        metrics={
            'f1': f1,
        },
        epochs=11,
        log_prefix='vgg_16_epoch_10'
    )

    model = Vgg16GAP('voc_classification_epoch_15', voc2012.get_class_count() -1)
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=16, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=16, shuffle=False, num_workers=6)
        },
        metrics={
            'f1': f1,
        },
        epochs=16,
        log_prefix='vgg_16_epoch_15'
    )

    model = Vgg16GAP('voc_classification_epoch_20', voc2012.get_class_count() -1)
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCClassification('train'), batch_size=16, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCClassification('val'), batch_size=16, shuffle=False, num_workers=6)
        },
        metrics={
            'f1': f1,
        },
        epochs=21,
        log_prefix='vgg_16_epoch_20'
    )