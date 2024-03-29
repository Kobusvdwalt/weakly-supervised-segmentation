def train_model(dataloaders, model, num_epochs):
    step = 0
    for epoch in range(num_epochs):
        model.event({
            'name': 'epoch_start',
            'epoch': epoch
        })

        for key in dataloaders.keys():
            model.event({
                'name': 'phase_start',
                'phase': key,
            })

            for batch_no, batch in enumerate(dataloaders[key]):
                step +=1
                model.event({
                    'name': 'minibatch',
                    'inputs': batch[0],
                    'labels': batch[1],
                    'data': batch[2],
                    'epoch': epoch+1,
                    'phase': key,
                    'batch': batch_no+1,
                    'step': step
                })

            model.event({
                'name': 'phase_end',
                'epoch': epoch,
                'phase': key,
                'batch': batch_no+1
            })

        model.event({
            'name': 'epoch_end',
            'epoch': epoch+1,
            'batch': batch_no+1
        })

def train(model, dataloaders, epochs = 15):
    model.to(model.device)
    train_model(dataloaders, model, epochs)
