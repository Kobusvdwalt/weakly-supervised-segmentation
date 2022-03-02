def start():
    import numpy as np
    import cv2, wandb
    from artifacts.artifact_manager import artifact_manager
    from torch.utils.data.dataloader import DataLoader
    from models.wass import WASS
    from models.unet import UNet
    from models.fiiyc import FIIYC
    from models.vgg16 import Vgg16GAP
    from training.train import train
    from data.loader_segmentation import VOCSegmentation

    artifact_manager.setArtifactContainer('CAM')
    model = Vgg16GAP()
    # wandb.init(entity='kobus_wits', project='wass_adv', name='WASS_vgg16_cam')
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(VOCSegmentation('train', dataset='voc'), batch_size=32, shuffle=True, num_workers=8),
    #         'val': DataLoader(VOCSegmentation('val', dataset='voc'), batch_size=32, shuffle=False, num_workers=8)
    #     },
    #     epochs=10,
    #     validation_mod=15000
    # )
    # wandb.finish()


    # artifact_manager.setArtifactContainer('WASS_unet')
    # model = UNet()
    # wandb.init(entity='kobus_wits', project='wass_adv', name='WASS_unet')
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(VOCSegmentation('train', dataset='voco'), batch_size=16, shuffle=True, num_workers=8),
    #         'val': DataLoader(VOCSegmentation('val', dataset='voco'), batch_size=16, shuffle=False, num_workers=8)
    #     },
    #     epochs=1,
    #     validation_mod=15000
    # )
    # wandb.finish()

    artifact_manager.setArtifactContainer('WASS_base')
    model = WASS()
    wandb.init(entity='kobus_wits', project='wass_adv', name='WASS_base')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(VOCSegmentation('train', dataset='voco'), batch_size=16, shuffle=True, num_workers=8),
            'val': DataLoader(VOCSegmentation('val', dataset='voco'), batch_size=16, shuffle=False, num_workers=8)
        },
        epochs=1000,
        validation_mod=15000
    )
    wandb.finish()

    # artifact_manager.setArtifactContainer('FIIYC')
    # model = FIIYC()
    # wandb.init(entity='kobus_wits', project='wass_adv', name='FIIYC')
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(VOCSegmentation('train', dataset='voco'), batch_size=16, shuffle=True, num_workers=8),
    #         'val': DataLoader(VOCSegmentation('val', dataset='voco'), batch_size=16, shuffle=False, num_workers=8)
    #     },
    #     epochs=1,
    #     validation_mod=15000
    # )
    # wandb.finish()




    # Video creation helper
    # *******************************
    # artifact_manager.setArtifactContainer('wass')
    # for image_no in range(0, 31):
    #     for type in ['mask_og', 'erase_og']:
    #         step_no = 0
    #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #         out = cv2.VideoWriter(artifact_manager.getDir() +f'/vid_{type}_{image_no}.avi', fourcc, 10.0, (256,  256))
    #         while True:
    #             image = cv2.imread(artifact_manager.getDir() + f'/{type}_{image_no}_{step_no}.png')
    #             if image is None:
    #                 print(f'image not found {image_no}:{step_no}')
    #                 break

    #             out.write(image)
    #             cv2.imshow('image', image)

    #             cv2.waitKey(1)
    #             step_no += 1

    #         out.release()