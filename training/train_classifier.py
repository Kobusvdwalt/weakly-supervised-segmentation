from models.get_model import get_model

def train_classifier(
    dataset_root,
    model_name = 'unet',
    epochs=51,
    batch_size=16,
    image_size=256,
):
    print('Model training : ', locals())
    from training.train import train
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation

    model = get_model(model_name)

    train(
        model=model,
        dataloaders = {
            'train': DataLoader(
                Segmentation(
                    dataset_root,
                    source='train',
                    source_augmentation='train',
                    image_size=image_size
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=6,
                pin_memory=True,
                prefetch_factor=8
            ),
        },
        epochs=epochs,
        validation_mod=10
    )
