def start():
    import numpy as np
    import cv2
    from artifacts.artifact_manager import artifact_manager
    from torch.utils.data.dataloader import DataLoader
    from models.wass import WASS
    from training.train import train
    from data.loader_segmentation import VOCSegmentation

    artifact_manager.setArtifactContainer('wass')
    model_base = WASS()
    model = model_base.new_instance()
    # wandb.init(entity='kobus_wits', project='wass_adv', name='adv')
    train(
        model=model,
        dataloaders = {
            'train': DataLoader(VOCSegmentation('train', dataset='voco'), batch_size=32, shuffle=True, num_workers=8),
            'val': DataLoader(VOCSegmentation('val', dataset='voco'), batch_size=32, shuffle=False, num_workers=8)
        },
        epochs=15000,
        validation_mod=15000
    )
    # wandb.finish()

    # artifact_manager.setArtifactContainer('wass')

    for image_no in range(0, 31):
        for type in ['mask_og', 'erase_og']:
            step_no = 0
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(artifact_manager.getDir() +f'/vid_{type}_{image_no}.avi', fourcc, 10.0, (256,  256))
            while True:
                image = cv2.imread(artifact_manager.getDir() + f'/{type}_{image_no}_{step_no}.png')
                if image is None:
                    print(f'image not found {image_no}:{step_no}')
                    break

                out.write(image)
                cv2.imshow('image', image)

                cv2.waitKey(1)
                step_no += 1

            out.release()