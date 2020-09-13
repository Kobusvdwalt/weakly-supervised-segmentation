


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))

    from metrics.f1 import f1
    from metrics.accuracy import accuracy

    from models.model_factory import Datasets, Models, get_model
    from models import model_factory
    from torch.utils.data import DataLoader
    from data.loader_factory import get_loader, LoaderSplit, LoaderType
    from torch.optim import lr_scheduler
    from torchvision import datasets, models, transforms

    import torch
    import numpy as np
    import time
    import os

    def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs, metrics):
        print('Training Start: ' + str(time.time()))

        metric_best = 0.0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    if (epoch % 5 != 0):
                        break
                    model.eval()

                batch_count = 0
                metric_store = {}
                for metric_name in metrics:
                    metric_store[metric_name] = 0
                for inputs, labels, _, _, _ in dataloaders[phase]:
                    batch_count += 1

                    inputs = inputs.permute(0, 3, 1, 2)
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).float()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        for metric_name in metrics:
                            metric_func = metrics[metric_name]
                            metric_result = metric_func(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
                            metric_store[metric_name] += metric_result

                    print('{} Batch: {} '.format(phase, batch_count), end='')
                    for metric_name in metric_store:
                        print(' {} {:.4f},'.format(metric_name, metric_store[metric_name] / batch_count), end='')
                    print('', end='\r')
                print('')

                key = None
                metric_epoch = metric_store[list(metric_store)[0]] / batch_count
                if phase == 'train':
                    scheduler.step()

                if phase == 'val' and metric_epoch > metric_best:
                    metric_best = metric_epoch
                    model.save()

    # #############################################
    # Config training here :
    dataset = Datasets.voc2012
    model = Models.Unet
    metrics = {
        'f1': f1,
        'accuracy': accuracy,
    }
    epochs = 15
    batch_size=4
    # #############################################

    # Set up datasets
    dataset_train = get_loader(dataset, LoaderType.segmentation, LoaderSplit.train)
    dataset_val = get_loader(dataset, LoaderType.segmentation, LoaderSplit.val)
    
    # Set up dataloaders
    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6),
        'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=6)
    }

    # Set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(dataset, model)
    model.to(device)

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Kick off training
    train_model(dataloaders, model, torch.nn.BCELoss(), optimizer, scheduler, epochs, metrics)