def start():
    import json, torch, os
    import numpy as np
    import matplotlib.pyplot as plt

    from artifacts.artifact_manager import artifact_manager
    from training._common import move_to
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.vgg16_blob import Vgg16GAPBlob
    from data.voc2012_loader_segmentation import PascalVOCSegmentation

    blob_sizes = [1, 9, 27, 51]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # # Train Baseline
    # model = Vgg16GAPBlob(name=f'voc_vgg16_blob_base', blob_size=0)
    # train(
    #     model=model,
    #     dataloaders = {
    #         'train': DataLoader(PascalVOCSegmentation('train'), batch_size=32, shuffle=True, num_workers=6),
    #         'val': DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
    #     },
    #     epochs=21,
    # )

    # # Train Erased
    # for blob_size in blob_sizes:
    #     model = Vgg16GAPBlob(name=f'voc_vgg16_blob_{blob_size}', blob_size=blob_size)
    #     train(
    #         model=model,
    #         dataloaders = {
    #             'train': DataLoader(PascalVOCSegmentation('train'), batch_size=32, shuffle=True, num_workers=6),
    #             'val': DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
    #         },
    #         epochs=21,
    #     )

    # # Plot train/val data
    # def get_metric(file_path, metric_key, phase='train'):
    #     json_file = open(artifact_manager.getDir() + file_path)
    #     data = json.load(json_file)

    #     metric = []
    #     epoch = []

    #     for entry in data['entries']:
    #         if entry['phase'] != phase:
    #             continue
    #         metric.append(entry[metric_key])
    #         epoch.append(entry['epoch'])

    #     return metric, epoch
    
    # def plot_metric(metric_key):
    #     plt.clf()
    #     f1, epoch = get_metric(f'voc_vgg16_blob_base_training_log.json', metric_key, 'train')
    #     plt.plot(epoch, f1, linestyle='--', color=colors[0], label='Original Train')

    #     f1, epoch = get_metric(f'voc_vgg16_blob_base_training_log.json', metric_key, 'val')
    #     plt.plot(epoch, f1, linestyle='-', color=colors[0], marker='s', label='Original Val')

    #     for blob_index, blob_size in enumerate(blob_sizes):
    #         f1, epoch = get_metric(f'voc_vgg16_blob_{blob_size}_training_log.json', metric_key,'train')
    #         plt.plot(epoch, f1, linestyle='--', color=colors[blob_index+1], label=None)

    #         f1, epoch = get_metric(f'voc_vgg16_blob_{blob_size}_training_log.json', metric_key, 'val')
    #         plt.plot(epoch, f1, linestyle='-', color=colors[blob_index+1], marker='s', label=f'Erased_{blob_size}')

    #     plt.title(f'{metric_key} over time')
    #     plt.legend()
    #     plt.xticks(epoch)
    #     plt.savefig(fname=artifact_manager.getDir() + f'{metric_key}.jpg')

    # plot_metric('f1')
    # plot_metric('loss')

    # Visualize images and masks
    blob_image = []
    blob_erase = []
    masks_input = []
    masks_expand = []
    masks_shrink = []
    masks_expand_shrink = []

    for blob_size in blob_sizes:
        data = DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
        model = Vgg16GAPBlob(name=f'voc_vgg16_blob_{blob_size}', blob_size=blob_size)

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
        ncol = len(blob_sizes) + 1

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
        for m in range(0, len(blob_sizes)):
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
        
        for m in range(0, len(blob_sizes)):
            axes[0].imshow(blob_image[0][i])
            axes[1].imshow(masks_input[0][i])
            axes[2].imshow(masks_expand[m][i])
            axes[3].imshow(masks_expand_shrink[m][i])

            plt.savefig(fname=artifact_manager.getDir() + f'vis2/sample_{i}_{m}.jpg')

    # TODO:
    # * We can set up another graph showing the number of pixels at each blob level
    # * Showing the best and worse performing classes might provide additional insight to shape learning
