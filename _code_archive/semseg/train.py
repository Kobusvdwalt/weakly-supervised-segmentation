if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data import PascalVOCSegmentation
    from unet import unet

    import torch
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import numpy as np
    import torchvision
    from torchvision import datasets, models, transforms
    import matplotlib.pyplot as plt
    import time
    import os
    import copy

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up dataloader
    pascal_train = PascalVOCSegmentation(source='train')
    pascal_val = PascalVOCSegmentation(source='val')

    dataloaders = {
        'train': DataLoader(pascal_train, batch_size=8, shuffle=True, num_workers=6),
        'val': DataLoader(pascal_val, batch_size=8, shuffle=False, num_workers=6)
    }

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    if (epoch % 5 != 0):
                        break
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                count = 0
                for inputs, labels in dataloaders[phase]:
                    count += 1

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

                        _, preds = torch.max(outputs, 1)
                        _, labels_pred = torch.max(labels, 1)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels_pred)

                    epoch_loss = running_loss / count
                    epoch_acc = running_corrects.double() / count
 
                    print('{} batch: {} Loss: {:.4f} Acc: {:.4f}.............'.format(phase, count, epoch_loss, epoch_acc), end='\r')

                if phase == 'train':
                    scheduler.step()

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), './checkpoints/unet.pt')

    unet.to(device)

    optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    train_model(unet, torch.nn.BCELoss(), optimizer, scheduler)