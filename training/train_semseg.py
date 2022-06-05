
from models.get_model import get_model
from training.config_manager import Config

def train_semseg(config: Config):
    config_json = config.toDictionary()
    print('train_semseg')
    print(config_json)
    from training.train import train
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager
    import wandb
    import os

    model = get_model(config.semseg_name)
    wandb.init(entity='kobus_wits', project='wass_semseg', name=config.sweep_id + '_s_' + config.semseg_name, config=config_json)
    wandb.watch(model)
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(
                Segmentation(
                    config.semseg_dataset_root,
                    source='train',
                    augmentation='train',
                    image_size=config.semseg_image_size,
                    requested_labels=['segmentation', 'pseudo'],
                    pseudo_root=os.path.join(artifact_manager.getDir(), 'labels_rw')
                ),
                batch_size=config.semseg_batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2
            ),
        },
        epochs=config.semseg_epochs,
        validation_mod=10
    )
    wandb.finish()
