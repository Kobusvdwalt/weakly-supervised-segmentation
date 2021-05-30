def start():
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.vgg16 import Vgg16GAP
    from training._common import Checkpointer, Visualizer
    
    from data.voc2012_loader_classification import PascalVOCClassification
    from data.voc2012_loader_segmentation import PascalVOCSegmentation

    # VGG16
    model = Vgg16GAP(name='voc_vgg16', blob_size=1, event_consumers=[Visualizer()])
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentation('train'), batch_size=32, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
        },
        epochs=21,
    )

    model = Vgg16GAP(name='voc_vgg16', blob_size=9, event_consumers=[Visualizer()])
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentation('train'), batch_size=32, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
        },
        epochs=21,
    )

    model = Vgg16GAP(name='voc_vgg16', blob_size=27, event_consumers=[Visualizer()])
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentation('train'), batch_size=32, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
        },
        epochs=21,
    )

    model = Vgg16GAP(name='voc_vgg16', blob_size=51, event_consumers=[Visualizer()])
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(PascalVOCSegmentation('train'), batch_size=32, shuffle=True, num_workers=6),
            'val': DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
        },
        epochs=21,
    )