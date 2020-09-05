if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))

    from torch.utils.data import DataLoader
    from data.loaders import PascalVOCClassificationMulticlass
    from models.vgg_gap_up_p import Vgg16

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
    pascal_train = PascalVOCClassificationMulticlass(source='train')
    pascal_val = PascalVOCClassificationMulticlass(source='val')

    dataloaders = {
        'train': DataLoader(pascal_train, batch_size=16, shuffle=True, num_workers=6),
        'val': DataLoader(pascal_val, batch_size=16, shuffle=False, num_workers=6)
    }

    def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
        since = time.time()

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

                tp = torch.zeros(20).to(device)
                tn = torch.zeros(20).to(device)
                fp = torch.zeros(20).to(device)
                fn = torch.zeros(20).to(device)

                batch_count = 0
                for inputs, labels, _ in dataloaders[phase]:
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

                        outputs[outputs > 0.5] = 1
                        outputs[outputs <= 0.5] = 0

                        labels[labels > 0.5] = 1
                        labels[labels <= 0.5] = 0

                        tp += torch.sum(torch.logical_and(outputs == 1, labels == 1), 0)
                        fp += torch.sum(torch.logical_and(outputs == 1, labels == 0), 0)
                        fn += torch.sum(torch.logical_and(outputs == 0, labels == 1), 0)
                        tn += torch.sum(torch.logical_and(outputs == 0, labels == 0), 0)

                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp + 1e-7)
                    recall = tp / (tp + fn + 1e-7)
                    f1 = 2 * ((precision * recall ) / (precision + recall + 1e-7))

                    accuracy_np = accuracy.cpu().numpy()
                    precision_np = precision.cpu().numpy()
                    recall_np = recall.cpu().numpy()
                    f1_np = f1.cpu().numpy()

                    accuracy_m = np.sum(accuracy_np) / 20
                    precision_m = np.sum(precision_np) / 20
                    recall_m = np.sum(recall_np) / 20
                    f1_m = np.sum(f1_np) / 20

                    print('{} Batch: {} Accuracy: {:.4f} Precision {:.4f} Recall {:.4f} F1 {:.4f}.............'.format(phase, batch_count, accuracy_m, precision_m, recall_m, f1_m), end='\r')
                    epoch_acc = accuracy_m

                print('{} Batch: {} Accuracy: {:.4f} Precision {:.4f} Recall {:.4f} F1 {:.4f}.............'.format(phase, batch_count, accuracy_m, precision_m, recall_m, f1_m))
                print('accuracy_np:')
                print(accuracy_np)

                print('precision_np:')
                print(precision_np)

                print('recall_np:')
                print(recall_np)

                print('f1_np:')
                print(f1_np)

                if phase == 'train':
                    scheduler.step()

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    model.save()

    vgg = Vgg16('multiclass_up', 20)
    vgg.to(device)

    optimizer = torch.optim.Adam(vgg.parameters(), lr=0.0002)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    train_model(vgg, torch.nn.BCELoss(), optimizer, scheduler)