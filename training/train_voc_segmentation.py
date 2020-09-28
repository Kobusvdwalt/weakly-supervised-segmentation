if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))

    from training.train import train
    from metrics.f1 import f1
    from metrics.iou import iou
    from models.model_factory import Datasets, Models
    from data.loader_factory import LoaderType

    train(
        dataset=Datasets.voc2012,
        loader_type=LoaderType.segmentation,
        model=Models.Unet,
        metrics={
            'f1': f1,
            'iou': iou
        },
        epochs=100,
        batch_size=4,
    )