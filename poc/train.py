if __name__ == '__main__':
    import torch, cv2
    import numpy as np

    from torch.utils.data.dataloader import DataLoader
    from data_loader import VOCOClassification
    from classifier import Vgg16GAP
    from adversary import Adversary

    # Set up our train/val dataloaders
    dataloader = DataLoader(VOCOClassification('train'), batch_size=32, shuffle=True, num_workers=8)

    # Other set up
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 30

    # Instantiate a new classifier
    model_classifer = Vgg16GAP(class_count=20)
    model_classifer.train()
    model_classifer.to(device)

    # Instantiate a new adversary
    model_adversary = Adversary(class_count=21) # +1 class for background
    model_adversary.train()
    model_adversary.to(device)

    # Start training loop
    for epoch in range(num_epochs):
        for batch_no, batch in enumerate(dataloader):
            inputs_in = batch[0]
            labels_in = batch[1]

            inputs = inputs_in.to(device).float()
            labels = labels_in.to(device).float()

            
            adversary_output = model_adversary.training_pass(inputs, labels)
            if (batch_no % 3 == 0):
                classification = model_classifer.training_pass(inputs)
            else:
                classification = model_classifer.training_pass(adversary_output['erase_image'])

            if batch_no % 2 == 0:
            # Train classifier
                c_loss_bce = model_classifer.loss_bce(classification, labels)
                c_loss_bce.backward()
                model_classifer.optimizer.step()
            else:
            # Train adversary
                # Channel loss
                a_loss_bce_c = model_adversary.loss_bce(adversary_output['classification'], labels)

                # Erase loss
                a_loss_m = torch.mean(classification[labels > 0.5])

                # Constrain loss
                a_loss_constrain = torch.mean(adversary_output['erase_mask']) * 0.1

                # Final loss
                a_loss_final = a_loss_constrain + a_loss_m + a_loss_bce_c

                a_loss_final.backward()
                model_adversary.optimizer.step()

            model_classifer.optimizer.zero_grad()
            model_adversary.optimizer.zero_grad()

            cv2.imshow('input', np.moveaxis(inputs[0].clone().detach().cpu().numpy(), 0, -1))
            cv2.imshow('erase_image', np.moveaxis(adversary_output['erase_image'][0].clone().detach().cpu().numpy(), 0, -1))
            cv2.imshow('erase_mask', np.moveaxis(adversary_output['erase_mask'][0].clone().detach().cpu().numpy(), 0, -1))
            cv2.waitKey(1)