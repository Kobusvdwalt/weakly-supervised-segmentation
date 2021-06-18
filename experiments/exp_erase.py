def start():
    import torch, os, cv2
    import numpy as np
    import matplotlib.pyplot as plt

    from artifacts.artifact_manager import artifact_manager
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from training._common import get_metric
    from models.vgg16 import Vgg16GAP
    from data.loader_erase import VOCErase

    erase_sizes = [9, 27, 51]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Config
    artifact_manager.setArtifactContainer('erase_vgg16_voc')
    model_base = Vgg16GAP(20)
    epochs = 13
    validation_mod = 3

    # # Train Baseline
    # model_base.name = 'base'
    # model = model_base.new_instance()
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(VOCErase('train', type='none'), batch_size=32, shuffle=True, num_workers=8),
    #         'val': DataLoader(VOCErase('val', type='none'), batch_size=32, shuffle=False, num_workers=8)
    #     },
    #     epochs=epochs,
    #     validation_mod=validation_mod
    # )

    # # Train Bounding Box
    # model_base.name = 'bbox'
    # model = model_base.new_instance()
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(VOCErase('train', type='erase_bbox'), batch_size=32, shuffle=True, num_workers=8),
    #         'val': DataLoader(VOCErase('val', type='erase_bbox'), batch_size=32, shuffle=False, num_workers=8)
    #     },
    #     epochs=epochs,
    #     validation_mod=validation_mod
    # )

    # # Train Erased
    # for erase_size in erase_sizes:
    #     model_base.name = f'erase_{erase_size}'
    #     model = model_base.new_instance()
    #     train(
    #         model=model,
    #         dataloaders = {
    #             'train': DataLoader(VOCErase('train', type='erase_gaus', size=erase_size), batch_size=32, shuffle=True, num_workers=8),
    #             'val': DataLoader(VOCErase('val', type='erase_gaus', size=erase_size), batch_size=32, shuffle=False, num_workers=8)
    #         },
    #         epochs=epochs,
    #         validation_mod=validation_mod
    #     )

    # # Train BBox no context
    # model_base.name = 'bbnc'
    # model = model_base.new_instance()
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(VOCErase('train', type='erase_bbnc'), batch_size=32, shuffle=True, num_workers=8),
    #         'val': DataLoader(VOCErase('val', type='erase_bbnc'), batch_size=32, shuffle=False, num_workers=8)
    #     },
    #     epochs=epochs,
    #     validation_mod=validation_mod
    # )

    # # Plot train/val data
    # def plot_metric(metric_key):
    #     plt.clf()
    #     f1, epoch = get_metric(f'base_training_log.json', metric_key, 'train')
    #     plt.plot(epoch, f1, linestyle='--', color=colors[0], label='Original Train')

    #     f1, epoch = get_metric(f'base_training_log.json', metric_key, 'val')
    #     plt.plot(epoch, f1, linestyle='-', color=colors[0], marker='s', label='Original Val')

    #     f1, epoch = get_metric(f'bbox_training_log.json', metric_key, 'train')
    #     plt.plot(epoch, f1, linestyle='--', color=colors[1], label='BBox Train')

    #     f1, epoch = get_metric(f'bbox_training_log.json', metric_key, 'val')
    #     plt.plot(epoch, f1, linestyle='-', color=colors[1], marker='s', label='BBox Val')

    #     f1, epoch = get_metric(f'bbnc_training_log.json', metric_key, 'train')
    #     plt.plot(epoch, f1, linestyle='--', color=colors[2], label='BBnc Train')

    #     f1, epoch = get_metric(f'bbnc_training_log.json', metric_key, 'val')
    #     plt.plot(epoch, f1, linestyle='-', color=colors[2], marker='s', label='BBnc Val')

    #     for blob_index, erase_size in enumerate(erase_sizes):
    #         f1, epoch = get_metric(f'erase_{erase_size}_training_log.json', metric_key,'train')
    #         plt.plot(epoch, f1, linestyle='--', color=colors[blob_index+3], label=None)

    #         f1, epoch = get_metric(f'erase_{erase_size}_training_log.json', metric_key, 'val')
    #         plt.plot(epoch, f1, linestyle='-', color=colors[blob_index+3], marker='s', label=f'Erased_{erase_size}')

    #     plt.title(f'{metric_key} over time')
    #     plt.legend()
    #     plt.xticks(epoch)
    #     plt.savefig(fname=artifact_manager.getDir() + f'{metric_key}.jpg')

    # plot_metric('f1')
    # plot_metric('loss')

    def preview(ds):
        for i in range(0, 5):
            inputs, labels, dp = ds.__getitem__(i)
            image = np.moveaxis(inputs['image'], 0, -1)
            cv2.imshow('image', image)
            cv2.imwrite(artifact_manager.getDir() + ds.type + '_' + str(i) + '.jpg', image * 225.0)
            cv2.waitKey(1)
            
    preview(VOCErase(source='val', type='none'))
    preview(VOCErase(source='val', type='erase_gaus', size=9))
    preview(VOCErase(source='val', type='erase_gaus', size=27))
    preview(VOCErase(source='val', type='erase_gaus', size=51))
    preview(VOCErase(source='val', type='erase_bbox'))
    preview(VOCErase(source='val', type='erase_bbnc'))

    return

    # Visualize images and masks
    blob_image = []
    blob_erase = []
    masks_input = []
    masks_expand = []
    masks_shrink = []
    masks_expand_shrink = []

    for erase_size in erase_sizes:
        data = DataLoader(dataset_type('val', erase_size=erase_size), batch_size=32, shuffle=False, num_workers=0)
        model_base.name = f'voc_vgg16_erase_{erase_size}'
        model_base.erase_size = erase_size
        model = model_base.new_instance()

        for batch_index, batch in enumerate(data):
            inputs_in = move_to(batch[0], model.device)
            labels_in = move_to(batch[1], model.device)
            model.to(model.device)

            image = inputs_in['image']
            segme = labels_in['segmentation']

            blob_image.append(np.moveaxis(image.clone().detach().cpu().numpy(), 1, -1))

            mask, _ = torch.max(segme[:, 1:], dim=1, keepdim=True)

            mask_expand = model.blob.expand(mask)
            mask_shrink = model.blob.shrink(mask)
            mask_expand_shrink = model.blob(mask)

            masks_input.append(mask[:, 0].clone().detach().cpu().numpy())
            masks_expand.append(mask_expand[:, 0].clone().detach().cpu().numpy())
            masks_shrink.append(mask_shrink[:, 0].clone().detach().cpu().numpy())
            masks_expand_shrink.append(mask_expand_shrink[:, 0].clone().detach().cpu().numpy())

            image = image * (1 - mask_expand_shrink)
            blob_erase.append(np.moveaxis(image.clone().detach().cpu().numpy(), 1, -1))
            break

    if not os.path.exists(artifact_manager.getDir() +'/vis1'):
        os.makedirs(artifact_manager.getDir() +'/vis1')

    for i in range(0, blob_image[0].shape[0]):
        plt.clf()
        nrow = 2
        ncol = len(erase_sizes) + 1

        fig, axes = plt.subplots(
            nrow, ncol,
            gridspec_kw=dict(wspace=0.05, hspace=0.05,
                            top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                            left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
            figsize=(ncol + 1, nrow + 1),
            sharey='row', sharex='col', #  optionally
        )

        for r in axes:
            for ax in r:
                ax.axis('off')
        
        axes[0, 0].imshow(blob_image[0][i])
        for m in range(0, len(erase_sizes)):
            axes[0, m+1].imshow(masks_expand_shrink[m][i])            
            axes[1, m+1].imshow(blob_erase[m][i])
        plt.savefig(fname=artifact_manager.getDir() + f'vis1/sample_{i}.jpg')

    if not os.path.exists(artifact_manager.getDir() +'/vis2'):
        os.makedirs(artifact_manager.getDir() +'/vis2')

    for i in range(0, blob_image[0].shape[0]):
        plt.clf()
        nrow = 1
        ncol = 4

        fig, axes = plt.subplots(
            nrow, ncol,
            gridspec_kw=dict(wspace=0.05, hspace=0.05,
                            top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                            left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
            figsize=(ncol + 1, nrow + 1),
            sharey='row', sharex='col', #  optionally
        )

        for ax in axes:
            ax.axis('off')
        
        for m in range(0, len(erase_sizes)):
            axes[0].imshow(blob_image[0][i])
            axes[1].imshow(masks_input[0][i])
            axes[2].imshow(masks_expand[m][i])
            axes[3].imshow(masks_expand_shrink[m][i])

            plt.savefig(fname=artifact_manager.getDir() + f'vis2/sample_{i}_{m}.jpg')

    masks_expand_pixel_count = []
    for masks in masks_expand:
        masks_expand_pixel_count.append(np.mean(masks))
    
    masks_expand_shrink_pixel_count = []
    for masks in masks_expand_shrink:
        masks_expand_shrink_pixel_count.append(np.mean(masks))

    plt.clf()
    plt.title(f'Mask Pixel Count per blur size')

    plt.plot(masks_expand_pixel_count)
    plt.plot(masks_expand_shrink_pixel_count)
    erase_sizes_str = [str(erase_size) for erase_size in erase_sizes]
    erase_sizes_idx = [i for i in range(0, len(erase_sizes))]

    plt.xticks(erase_sizes_idx, erase_sizes_str)
    plt.savefig(fname=artifact_manager.getDir() + f'expand.jpg')

    # TODO:
    # * Showing the best and worse performing classes might provide additional insight to shape learning
    # * Showing cams
    # * Plot of final val scores over blur size
    # * Would be interesting to see what edges do
    # * Two other options are different random images for each segmentation mask and random colors or random grey scales for different objects this way we retain seperation

