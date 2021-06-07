import torch
from training._common import move_to

def train_model(dataloaders, model, num_epochs):
    for epoch in range(num_epochs):
        model.event({
            'name': 'epoch_start',
            'epoch': epoch
        })

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            if phase == 'val':
                if epoch < 4 or epoch % 4 != 0:
                    break
                model.eval()
            
            model.event({
                'name': 'phase_start',
                'phase': phase,
            })

            for batch_no, batch in enumerate(dataloaders[phase]):
                inputs_in = batch[0]
                labels_in = batch[1]

                inputs = move_to(inputs_in, model.device)
                labels = move_to(labels_in, model.device)
                
                torch.set_grad_enabled(phase == 'train')

                # Model minibatch pass
                model.event({
                    'name': 'minibatch',
                    'inputs': inputs,
                    'labels': labels,
                    'epoch': epoch,
                    'phase': phase,
                    'batch': batch_no+1
                })

            model.event({
                'name': 'phase_end',
                'epoch': epoch,
                'phase': phase,
                'batch': batch_no+1
            })

        model.event({
            'name': 'epoch_end',
            'epoch': epoch,
            'batch': batch_no+1
        })

def train(model, dataloaders, epochs = 15):
    model.to(model.device)
    train_model(dataloaders, model, epochs)