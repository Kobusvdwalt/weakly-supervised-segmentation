from time import time
from data.voc2012 import renormalize
from models.erase_vgg16 import EraseVGG16
from training.config_manager import Config

def start():
    import torch, os, cv2, wandb, random
    import numpy as np
    import matplotlib.pyplot as plt

    from artifacts.artifact_manager import artifact_manager
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.vgg16 import Vgg16GAP
    from data.loader_erase import VOCErase

    
    # # Visualize images
    # artifact_manager.setArtifactContainer('erase_images')
    # def write_images(type, size=0):
    #     torch.backends.cudnn.deterministic = True
    #     random.seed(1)
    #     torch.manual_seed(1)
    #     torch.cuda.manual_seed(1)
    #     np.random.seed(1)
    #     dataset = VOCErase(dataset_root='datasets/generated/voc', source='val', type=type, size=size)
    #     dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    #     images = next(iter(dataloader))[0]['image'].clone().detach().cpu().numpy()
    #     for i, image in enumerate(images):
    #         image = np.moveaxis(image, 0, -1)
    #         cv2.imwrite(artifact_manager.getDir() + f'/{i}_{type}_{size}.png', renormalize(image) * 255)
    
    # write_images('none')
    # write_images('mask_base', size=1)
    # write_images('erase_bbox')
    # write_images('erase_bbnc')
    # write_images('erase_gaus', size=1)
    # write_images('erase_gaus', size=9)
    # write_images('erase_gaus', size=27)
    # write_images('erase_gaus', size=53)

    exp_start_time = str(time())

    sweeps = [
        Config({
            Config.KEY_ERASE_SWEEP_ID: f'{exp_start_time}_erase_base',
            Config.KEY_ERASE_TRAIN_DATASET: f'datasets/generated/voc_aug',
            Config.KEY_ERASE_EVAL_DATASET: f'datasets/generated/voc',
            Config.KEY_ERASE_MODE: f'none',
            Config.KEY_ERASE_MODEL: f'vgg16',
            Config.KEY_ERASE_BATCH_SIZE: 32,
            Config.KEY_ERASE_EPOCHS: 128
        }),
    ]

    for sweep_index, sweep in enumerate(sweeps):
        artifact_manager.setArtifactContainer(sweep.getValue(Config.KEY_ERASE_SWEEP_ID))

        # TODO: Pick different model based on sweep
        if sweep.getValue(Config.KEY_ERASE_MODEL) == 'vgg16':
            model = EraseVGG16()

        wandb.init(entity='kobus_wits', project='wass', name=sweep.getValue(Config.KEY_ERASE_SWEEP_ID))
        wandb.watch(model)
        train(
            model=model,
            dataloaders = {
                'train': DataLoader(VOCErase(
                        dataset_root=sweep.getValue(Config.KEY_ERASE_TRAIN_DATASET),
                        source='train',
                        type=sweep.getValue(Config.KEY_ERASE_MODE),
                        size=1
                    ),
                    batch_size=sweep.getValue(Config.KEY_ERASE_BATCH_SIZE),
                    shuffle=True,
                    num_workers=8
                ),
                # 'val': DataLoader(VOCErase(
                #         dataset_root=sweep.classifier_dataset_root,
                #         source='val',
                #         type=sweep.classifier_erase_type,
                #         size=sweep.classifier_erase_strength
                #     ),
                #     batch_size=sweep.classifier_batch_size_train,
                #     shuffle=False,
                #     num_workers=8
                # )
            },
            epochs=sweep.getValue(Config.KEY_ERASE_EPOCHS),
        )
        wandb.finish()
