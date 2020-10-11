if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))
    
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from metrics.f1 import f1
    from data import voc2012
    from models.vgg16_gap_feat_unfreeze import Vgg16GAPUnfreeze
    from data.voc2012_loaders import PascalVOCClassification


    unfreeze = 1
    model = Vgg16GAPUnfreeze('voc_classification', voc2012.get_class_count() -1, unfreeze)
    # model.load()
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
        log_prefix='unfreeze_' + str(unfreeze)
    )