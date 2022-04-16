from models.get_model import get_model

def train_affinitynet(
    dataset_root,
    model_name = 'affinitynet',
    epochs=51,
    image_size=256,
    batch_size=16,
):
    print('Training affinitynet : ', locals())
    from training.train import train
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager

    model = get_model(model_name)

    train(
        model=model,
        dataloaders = {
            'train': DataLoader(
                Segmentation(
                    dataset_root,
                    source='train',
                    source_augmentation='train',
                    image_size=image_size,
                    requested_labels=['affinity'],
                    affinity_root=artifact_manager.getDir()
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            ),
        },
        epochs=epochs,
        validation_mod=10
    )
