import sys, os, json, torch

sys.path.insert(0, os.path.abspath('../'))

from metrics.f1 import f1
from torch.optim import lr_scheduler
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device).float()
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")

def train_model(dataloaders, model, num_epochs, metrics, log_prefix):
    log = {}
    log['training_start'] = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log['model_name'] = model.name
    log['train'] = []
    log['val'] = []

    print('Training Start: ' + log['training_start'])

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
            for output_key in metrics:
                metric_store[output_key] = {}
                for metric_name in metrics[output_key]:
                    metric_store[output_key][metric_name] = 0

            for inputs_in, labels_in, data_package in dataloaders[phase]:
                batch_count += 1

                inputs = move_to(inputs_in, device)
                labels = move_to(labels_in, device)

                with torch.set_grad_enabled(phase == 'train'):
                    # Model forward pass
                    outputs = model(inputs)

                    # Record and store the metrics
                    for output_key in metrics:
                        for metric_name in metrics[output_key]:
                            metric_func = metrics[output_key][metric_name]
                            metric_result = metric_func(outputs[output_key].cpu().detach().numpy(), labels[output_key].cpu().detach().numpy())
                            metric_store[output_key][metric_name] += metric_result

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
            model.save()
            # TODO: This can be done with a touch more elegance
            # output_key = list(metric_store.keys())[0]
            # metric_key = list(metric_store[output_key].keys())[0]
            # metric_epoch = metric_store[output_key][metric_key] / batch_count
            # if phase == 'val' and metric_epoch > metric_best:
            #     metric_best = metric_epoch
            #     model.save()

            

def train(model, dataloaders, metrics = {'f1': f1}, epochs = 15, log_prefix=''):
    # Set up model
    model.to(device)

    # Kick off training
    train_model(dataloaders, model, epochs, metrics, log_prefix)