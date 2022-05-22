import wandb
from models.get_model import get_model
from training.config_manager import Config

def train_classifier(config: Config):
    config_json = config.toDictionary()
    print('train_classifier')
    print(config_json)
    from training.train import train
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation

    model = get_model(config.classifier_name)
    
    wandb.init(entity='kobus_wits', project='wass_classifier', name='classifier_' + config.classifier_name, config=config_json)
    wandb.watch(model)

    train(
        model=model,
        dataloaders = {
            'train': DataLoader(
                Segmentation(
                    config.classifier_dataset_root,
                    source='train',
                    augmentation='train',
                    image_size=config.classifier_image_size
                ),
                batch_size=config.classifier_batch_size_train,
                shuffle=True,
                num_workers=6,
                pin_memory=True,
                prefetch_factor=2
            ),
        },
        epochs=config.classifier_epochs,
        validation_mod=10
    )

    wandb.finish()
