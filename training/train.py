import sys, os

from torch.nn.modules.loss import BCELoss
sys.path.insert(0, os.path.abspath('../'))

from metrics.f1 import f1
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import torch
import numpy as np
import os
import json
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs, metrics, log_prefix):
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log = {}
    log['training_start'] = date_time
    log['model_name'] = model.name
    log['loss'] = str(criterion)
    log['optimizer'] = str(optimizer)
    log['train'] = []
    log['val'] = []    

    print('Training Start: ' + date_time)

    metric_best = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            if phase == 'val':
                if (epoch % 5 != 0):
                    break
                model.eval()

            batch_count = 0
            metric_store = {}
            for metric_name in metrics:
                metric_store[metric_name] = 0
            for inputs, labels, names, meta in dataloaders[phase]:
                batch_count += 1

                inputs = inputs.permute(0, 3, 1, 2)
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward(retain_graph=True)
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

            entry = {}
            entry['epoch'] = epoch
            for metric_name in metric_store:
                entry[metric_name] = metric_store[metric_name] / batch_count
            
            # Write logs
            date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            log['training_update'] = date_time
            log[phase].append(entry)
            with open('output/log__' + log_prefix + '__' + log['training_start'] + '.txt', 'w') as outfile:
                json.dump(log, outfile)

            if phase == 'train':
                scheduler.step()

            # Save model
            metric_epoch = metric_store[list(metric_store)[0]] / batch_count
            if phase == 'val' and metric_epoch > metric_best:
                metric_best = metric_epoch
                model.save()

            

def train(model, dataloaders, metrics = {'f1': f1}, loss = torch.nn.BCELoss(), epochs = 15, learning_rate = 1e-4, log_prefix=''):
    # Set up model
    model.to(device)

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # Kick off training
    train_model(dataloaders, model, loss, optimizer, scheduler, epochs, metrics, log_prefix)