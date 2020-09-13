if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))

    from training.train import train
    from metrics.f1 import f1
    from models.model_factory import Datasets, Models
    from data.loader_factory import LoaderType

    train(
        dataset=Datasets.voc2012,
        loader_type=LoaderType.classification,
        model=Models.Vgg16GAP,
        metrics={
            'f1': f1,
        },
        epochs=15,
        batch_size=16,
    )