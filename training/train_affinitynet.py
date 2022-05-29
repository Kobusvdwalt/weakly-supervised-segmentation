import wandb
from models.get_model import get_model
from training.config_manager import Config

def train_affinitynet(config: Config):
    config_json = config.toDictionary()
    print('train_affinitynet')
    print(config_json)
    from training.train import train
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager

    model = get_model(config.affinity_net_name)
    
    wandb.init(entity='kobus_wits', project='wass_affinity', name=config.sweep_id + '_a_' + config.affinity_net_name, config=config_json)
    wandb.watch(model)

    train(
        model=model,
        dataloaders = {
            'train': DataLoader(
                Segmentation(
                    config.classifier_dataset_root,
                    source='train',
                    augmentation='train',
                    image_size=config.affinity_net_image_size,
                    requested_labels=['affinity'],
                    affinity_root=artifact_manager.getDir()
                ),
                batch_size=config.affinity_net_batch_size,
                shuffle=False,
                pin_memory=False,
                num_workers=4,
                prefetch_factor=4
            ),
        },
        epochs=config.affinity_net_epochs,
        validation_mod=10
    )

    wandb.finish()
