

import sys, os
sys.path.insert(0, os.path.abspath('../'))

from metrics.f1 import f1
from metrics.accuracy import accuracy
from metrics.iou import iou
from data.cityscapes import label_to_image

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
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            for inputs, labels, names, widths, heights in dataloaders[phase]:
                batch_count += 1

                # image_np = inputs[0].numpy()
                # label_np = labels[0].numpy()
                # print(names[0])
                # cv2.imshow('image_np', image_np)
                # cv2.imshow('label_np', label_to_image(label_np))

                inputs = inputs.permute(0, 3, 1, 2)
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # output_np = outputs[0].cpu().detach().numpy()
                    # cv2.imshow('output_np', label_to_image(output_np))
                    # cv2.waitKey(1)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Record and store the metrics
                    for metric_name in metrics:
                        metric_func = metrics[metric_name]
                        metric_result = metric_func(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
                        metric_store[metric_name] += metric_result

                # Print feedback
                print('{} Batch: {} '.format(phase, batch_count), end='')
                for metric_name in metric_store:
                    print(' {} {:.4f},'.format(metric_name, metric_store[metric_name] / batch_count), end='')
                print('', end='\r')
            print('')

            metric_epoch = metric_store[list(metric_store)[0]] / batch_count
            if phase == 'train':
                scheduler.step()

            if phase == 'val' and metric_epoch > metric_best:
                metric_best = metric_epoch
                model.save()

def train(dataset = Datasets.cityscapes, loader_type=LoaderType.classification, model = Models.Unet, metrics = {'f1': f1}, epochs = 15, batch_size = 4, learning_rate = 1e-4):
    # Set up datasets
    dataset_train = get_loader(dataset, loader_type, LoaderSplit.train)
    dataset_val = get_loader(dataset, loader_type, LoaderSplit.val)

    # Set up dataloaders
    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6),
        'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=6)
    }

    # Set up model
    model = get_model(dataset, model)
    model.to(device)

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Kick off training
    train_model(dataloaders, model, torch.nn.BCELoss(), optimizer, scheduler, epochs, metrics)