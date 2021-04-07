import json, torch

from datetime import datetime
from training.helpers import move_to, NumpyEncoder
from artifacts.artifact_manager import artifact_manager

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(dataloaders, model, num_epochs):
    log = {}
    log['training_start'] = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log['model_name'] = model.name
    log['train'] = []
    log['val'] = []

    print('Training Start: ' + log['training_start'])

    metric_store_best = None
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        model.event({
            'name': 'EpochStart',
            'epoch': epoch
        })

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            if phase == 'val':
                if (epoch % 5 != 0):
                    break
                model.eval()
                
            batch_count = 0
            metric_store = None

            for inputs_in, labels_in, data_package in dataloaders[phase]:
                batch_count += 1

                inputs = move_to(inputs_in, device)
                labels = move_to(labels_in, device)
                
                torch.set_grad_enabled(phase == 'train')

                # Model forward and backward pass
                outputs = model(inputs)
                model.backward(outputs, labels)

                # Record and store the metrics
                metrics = model.metrics(outputs, labels)
                if metric_store is None:
                    metric_store = metrics
                else:
                    for output_key in metrics:
                        for metric_name in metrics[output_key]:
                            metric_store[output_key][metric_name] += metrics[output_key][metric_name]

                # Print feedback
                print('{} Batch: {} '.format(phase, batch_count), end='->')
                
                for output_key in metrics:
                    for metric_name in metric_store[output_key]:
                        if metric_name[0] != "_":
                            print(' {} {:.4f},'.format(output_key + '_' + metric_name, metric_store[output_key][metric_name] / batch_count), end='')
                print('', end='\r')
            print('')

            # Write logs
            entry = {}
            entry['epoch'] = epoch
            entry['outputs'] = {}

            for output_key in metric_store.keys():
                entry['outputs'][output_key] = {}
                for metric_name in metric_store[output_key]:
                    entry['outputs'][output_key][metric_name] = metric_store[output_key][metric_name] / batch_count
            
            log['training_update'] = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            log[phase].append(entry)
            with open(artifact_manager.getDir() + model.name + '_training_log.json', 'w') as outfile:
                json.dump(log, outfile, cls=NumpyEncoder)

            # Save model
            if phase == 'val':
                if metric_store_best is None:
                    metric_store_best = metric_store
                if model.should_save(metric_store_best, metric_store):
                    metric_store_best = metric_store
                    print('saving model')
                    model.save()

        model.event({
            'name': 'EpochEnd',
            'epoch': epoch
        })

def train(model, dataloaders, epochs = 15):
    model.to(device)
    train_model(dataloaders, model, epochs)