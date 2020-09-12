if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))

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

    def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs, output_size):
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

                tp = torch.zeros(output_size).to(device)
                tn = torch.zeros(output_size).to(device)
                fp = torch.zeros(output_size).to(device)
                fn = torch.zeros(output_size).to(device)

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

    # Config training here :
    dataset = Datasets.cityscapes
    model = Models.Vgg16GAP
    class_count = 19 # TODO: fix this class_count so that it's dataset dependant and not manual

    # Set up datasets
    dataset_train = get_loader(dataset, LoaderType.classification, LoaderSplit.train)
    dataset_val = get_loader(dataset, LoaderType.classification, LoaderSplit.val)
    
    # Set up dataloaders
    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=6),
        'val': DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=6)
    }

    # Set up model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(dataset, model)
    model.to(device)

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Kick off training
    train_model(dataloaders, model, torch.nn.BCELoss(), optimizer, scheduler, 40, class_count)