def start():
    import json, torch
    import numpy as np
    import matplotlib.pyplot as plt
    from artifacts.artifact_manager import artifact_manager
    from training._common import move_to
    from torch.utils.data.dataloader import DataLoader
    from training.train import train
    from models.vggm_blob import VggMBlob
    from data.voc2012_loader_segmentation import PascalVOCSegmentation

    blob_sizes = [1, 9, 27, 51]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for blob_size in blob_sizes:
        model = VggMBlob(name=f'voc_vggm_blob_{blob_size}', blob_size=blob_size)
        train(
            model=model,
            dataloaders = {
                'train': DataLoader(PascalVOCSegmentation('train'), batch_size=32, shuffle=True, num_workers=6),
                'val': DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
            },
            epochs=21,
        )

    # Plot training data
    def get_metric(file_path, metric_key, phase='train'):
        json_file = open(artifact_manager.getDir() + file_path)
        data = json.load(json_file)

        metric = []
        epoch = []

        for entry in data['entries']:
            if entry['phase'] != phase:
                continue
            metric.append(entry[metric_key])
            epoch.append(entry['epoch'])

        return metric, epoch
    
    def plot_metric(metric_key):
        plt.clf()
        for blob_index, blob_size in enumerate(blob_sizes):
            f1, epoch = get_metric(f'voc_vggm_blob_{blob_size}_training_log.json', metric_key,'train')
            plt.plot(epoch, f1, linestyle='--', color=colors[blob_index], label=None)

            f1, epoch = get_metric(f'voc_vggm_blob_{blob_size}_training_log.json', metric_key, 'val')
            plt.plot(epoch, f1, linestyle='-', color=colors[blob_index], marker='s', label=f'Erased_{blob_size}')

        plt.title(f'{metric_key} over time')
        plt.legend()
        plt.xticks(epoch)
        plt.savefig(fname=artifact_manager.getDir() + f'{metric_key}.jpg')

    plot_metric('f1')
    plot_metric('loss')

    # Visualize blobs
    blob_image = []
    blob_masks = []

    for blob_size in blob_sizes:
        data = DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
        model = VggMBlob(name=f'voc_vggm_blob_{blob_size}', blob_size=blob_size)

        for batch_index, batch in enumerate(data):
            inputs_in = move_to(batch[0], model.device)
            labels_in = move_to(batch[1], model.device)
            model.to(model.device)

            image = inputs_in['image']
            label = labels_in['classification']
            segme = labels_in['segmentation']

            blob_image.append(np.moveaxis(image.clone().detach().cpu().numpy(), 1, -1))

            mask, _ = torch.max(segme[:, 1:], dim=1, keepdim=True)
            mask = model.blob(mask)
            blob_masks.append(mask[:, 0].clone().detach().cpu().numpy())
            break

    for i in range(0, blob_image[0].shape[0]):
        plt.clf()
        nrow = 1
        ncol = len(blob_sizes) + 1

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
        
        axes[0].imshow(blob_image[0][i])
        for m in range(0, len(blob_sizes)):
            axes[m+1].imshow(blob_masks[m][i])
        plt.savefig(fname=artifact_manager.getDir() + f'sample_{i}.jpg')
