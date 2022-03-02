def start():
    import torch, os, cv2, wandb, random
    import numpy as np
    import matplotlib.pyplot as plt

    from artifacts.artifact_manager import artifact_manager
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.vgg16 import Vgg16GAP
    from data.loader_erase import VOCErase

    # Config
    artifact_manager.setArtifactContainer('erase_vgg16_voc')
    model_base = Vgg16GAP(20)
    epochs = 15
    validation_mod = 3

    # Visualize images
    # def write_images(type, size=0):
    #     torch.backends.cudnn.deterministic = True
    #     random.seed(1)
    #     torch.manual_seed(1)
    #     torch.cuda.manual_seed(1)
    #     np.random.seed(1)
    #     images = next(iter(DataLoader(VOCErase(source='val', type=type, size=size, dataset='voc'), batch_size=32, shuffle=True, num_workers=0)))[0]['image'].clone().detach().cpu().numpy()
    #     for i, image in enumerate(images):
    #         image = np.moveaxis(image, 0, -1)
    #         cv2.imwrite(artifact_manager.getDir() + f'/{type}_{size}_{i}.png', image * 255)
    
    # write_images('none')
    # write_images('mask_base', size=1)
    # write_images('erase_bbox')
    # write_images('erase_bbnc')
    # write_images('erase_gaus', size=1)
    # write_images('erase_gaus', size=9)
    # write_images('erase_gaus', size=27)
    # write_images('erase_gaus', size=53)

    # # Train Baseline
    wandb.init(entity='kobus_wits', project='wass', name='erase_base')
    model_base.name = 'base'
    model = model_base.new_instance()
    wandb.watch(model)
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(VOCErase('train', type='none'), batch_size=32, shuffle=True, num_workers=8),
            'val': DataLoader(VOCErase('val', type='none'), batch_size=32, shuffle=False, num_workers=8)
        },
        epochs=epochs,
        validation_mod=validation_mod
    )
    wandb.finish()

    # # Train Mask
    # wandb.init(entity='kobus_wits', project='wass', name='erase_omask')
    # model_base.name = 'base'
    # model = model_base.new_instance()
    # wandb.watch(model)
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(VOCErase('train', type='mask_base', size=1), batch_size=32, shuffle=True, num_workers=8),
    #         'val': DataLoader(VOCErase('val', type='mask_base', size=1), batch_size=32, shuffle=False, num_workers=8)
    #     },
    #     epochs=epochs,
    #     validation_mod=validation_mod
    # )
    # wandb.finish()

    # # Train Bounding Box
    # wandb.init(entity='kobus_wits', project='wass', name='erase_bbox')
    # model_base.name = 'bbox'
    # model = model_base.new_instance()
    # wandb.watch(model)
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(VOCErase('train', type='erase_bbox'), batch_size=32, shuffle=True, num_workers=8),
    #         'val': DataLoader(VOCErase('val', type='erase_bbox'), batch_size=32, shuffle=False, num_workers=8)
    #     },
    #     epochs=epochs,
    #     validation_mod=validation_mod
    # )
    # wandb.finish()

    # Train BBox no context
    wandb.init(entity='kobus_wits', project='wass', name='erase_bbnc')
    model_base.name = 'bbnc'
    model = model_base.new_instance()
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(VOCErase('train', type='erase_bbnc'), batch_size=32, shuffle=True, num_workers=8),
            'val': DataLoader(VOCErase('val', type='erase_bbnc'), batch_size=32, shuffle=False, num_workers=8)
        },
        epochs=epochs,
        validation_mod=validation_mod
    )
    wandb.finish()

    # Train Erased
    # erase_sizes = [1, 27, 53]
    # for erase_size in erase_sizes:
    #     wandb.init(entity='kobus_wits', project='wass', name=f'erase_{erase_size}')
    #     model_base.name = f'erase_{erase_size}'
    #     model = model_base.new_instance()
    #     wandb.watch(model)
    #     train(
    #         model=model,
    #         dataloaders = {
    #             'train': DataLoader(VOCErase('train', type='erase_gaus', size=erase_size), batch_size=32, shuffle=True, num_workers=8),
    #             'val': DataLoader(VOCErase('val', type='erase_gaus', size=erase_size), batch_size=32, shuffle=False, num_workers=8)
    #         },
    #         epochs=epochs,
    #         validation_mod=validation_mod
    #     )
    #     wandb.finish()
