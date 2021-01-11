import sys, os, json, torch

sys.path.insert(0, os.path.abspath('../'))

from datetime import datetime
from training.helpers import move_to

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
def train_model(dataloaders, model, num_epochs, log_prefix):
    log = {}
    log['training_start'] = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log['model_name'] = model.name
    log['train'] = []
    log['val'] = []

    print('Training Start: ' + log['training_start'])

    metric_store_best = None
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            if phase == 'val':
                if (epoch % 5 != 0):
                    break
                model.eval()

            model.epoch_start()

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
                        print(' {} {:.4f},'.format(output_key + '_' + metric_name, metric_store[output_key][metric_name] / batch_count), end='')
                print('', end='\r')
            print('')

            entry = {}
            entry['epoch'] = epoch
            for output_key in metric_store.keys():
                entry[output_key] = {}
                for metric_name in metric_store[output_key]:
                    entry[output_key][metric_name] = metric_store[output_key][metric_name] / batch_count
            
            # Write logs
            log['training_update'] = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            log[phase].append(entry)
            with open('output/log__' + log_prefix + '__' + log['training_start'] + '.txt', 'w') as outfile:
                json.dump(log, outfile)

            # Save model
            if phase == 'val':
                if metric_store_best is None:
                    metric_store_best = metric_store
                if model.should_save(metric_store_best, metric_store):
                    model.save()

def train(model, dataloaders, epochs = 15, log_prefix=''):
    # Set up model
    model.to(device)

    # Kick off training
    train_model(dataloaders, model, epochs, log_prefix)