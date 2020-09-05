if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data import PascalVOCSegmentation
    from unet import model

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
    import cv2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up dataloader
    pascal_train = PascalVOCSegmentation(source='train')
    pascal_val = PascalVOCSegmentation(source='val')

    dataloaders = {
        'train': DataLoader(pascal_train, batch_size=8, shuffle=True, num_workers=6),
        'val': DataLoader(pascal_val, batch_size=8, shuffle=False, num_workers=6)
    }

    def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 100.0

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
                    labels = labels.permute(0, 3, 1, 2)

                    inputs = inputs.to(device).float()
                    labels = labels.to(device).float()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        # Visualize
                        outputs_t = outputs.permute(0, 2, 3, 1)
                        outputs_np = outputs_t.cpu().detach().numpy()
                        out_0 = outputs_np[0]
                        cv2.imshow('out_0', out_0)

                        #inputs_t = inputs.permute(0, 2, 3, 1)
                        #inputs_np = inputs_t.cpu().detach().numpy()
                        #in_0 = inputs_np[0]
                        #cv2.imshow('in_0', in_0)

                        labels_t = labels.permute(0, 2, 3, 1)
                        labels_np = labels_t.cpu().detach().numpy()
                        lab_0 = labels_np[0]

                        cv2.imshow('lab_0', lab_0)
                        cv2.waitKey(2)

                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        _, preds = torch.max(outputs, 1)
                        _, labels_pred = torch.max(labels, 1)

                    running_loss += loss.item() * inputs.size(0)
                    epoch_loss = running_loss / count
 
                    print('{} batch: {} Loss: {:.4f} Acc: {:.4f}.............'.format(phase, count, epoch_loss, 0), end='\r')

                if phase == 'train':
                    scheduler.step()

                if phase == 'val' and epoch_loss < best_loss:
                    print('')
                    print('saving checkpoint')
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), './checkpoints/unet_8_16_n.pt')

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    train_model(model, torch.nn.BCELoss(), optimizer, scheduler)