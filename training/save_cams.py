import wandb
from numpy import average, float32
import torch
import torch.nn.functional as F
from data.voc2012 import label_to_image
from metrics.iou import iou
from models.get_model import get_model
from training.config_manager import Config

def save_cams(config: Config):
    config_json = config.toDictionary()
    print('train_semseg')
    print(config_json)
    import shutil
    import cv2
    import os
    import numpy as np
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager

    # Set up model
    model = get_model(config.classifier_name)
    model.load()
    model.to(model.device)

    # Set up data loader
    dataloader = DataLoader(
        Segmentation(
            config.classifier_dataset_root,
            source='train',
            augmentation='val',
            image_size=config.classifier_image_size,
            requested_labels=['classification', 'segmentation']
        ),
        batch_size=config.cams_batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=4
    )

    # Clear and create destination directory
    cam_path = os.path.join(artifact_manager.getDir(), 'cam')
    if (os.path.exists(cam_path)):
        shutil.rmtree(cam_path)
    os.makedirs(cam_path)

    label_cam_path = os.path.join(artifact_manager.getDir(), 'labels_cam')
    if (os.path.exists(label_cam_path)):
        shutil.rmtree(label_cam_path)
    os.makedirs(label_cam_path)

    for batch_no, batch in enumerate(dataloader):
        inputs_in = batch[0]
        labels_in = batch[1]
        datapacket_in = batch[2]

        # Run images through model and get raw cams
        with torch.no_grad():
            cams = model.event({
                'name': 'get_cam',
                'inputs': inputs_in,
                'labels': labels_in,
                'batch': batch_no+1
            })

        # Save out cams
        for cam_no, cam in enumerate(cams):
            # Save out ground truth labels for testing the rest of the system
            if config.cams_save_gt_labels:
                cam = labels_in['segmentation'][cam_no][1:]
                cam = F.adaptive_avg_pool2d(cam, [32, 32]).numpy()

                for i in range(0, cam.shape[0]):
                    cam[i] = cv2.blur(cam[i], (3, 3))
                    cam[i] = cv2.blur(cam[i], (3, 3))

            # Disregard false positives
            gt_mask = labels_in['classification'][cam_no].numpy()
            gt_mask[gt_mask > 0.5] = 1
            gt_mask[gt_mask <= 0.5] = 0
            gt_mask = np.expand_dims(np.expand_dims(gt_mask, -1), -1)
            cam *= gt_mask

            # Scale CAM to match input size
            cam = np.moveaxis(cam, 0, -1)
            cam = cv2.resize(cam, (config.classifier_image_size, config.classifier_image_size), interpolation=cv2.INTER_LINEAR)
            cam = np.moveaxis(cam, -1, 0)

            # - Cut CAM from input size and upscale to original image size
            width = datapacket_in['width'][cam_no].detach().numpy()
            height = datapacket_in['height'][cam_no].detach().numpy()
            content_width = datapacket_in['content_width'][cam_no].detach().numpy()
            content_height = datapacket_in['content_height'][cam_no].detach().numpy()
            cam = cam[:, 0:content_height, 0:content_width]
            cam = np.moveaxis(cam, 0, -1)
            cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
            cam = np.moveaxis(cam, -1, 0)

            # Normalize each cam map to between 0 and 1
            cam_max = np.max(cam, (1, 2), keepdims=True)
            cam_norm = cam / (cam_max + 1e-5)

            cam_bg = np.power(1 - np.max(cam_norm, axis=0, keepdims=True), 4)
            cam_with_bg = np.concatenate((cam_bg, cam_norm), axis=0)
            label_cam = label_to_image(cam_with_bg)

            # Collapse cam from 3d into long 2d
            cam_norm = np.reshape(cam_norm, (cam_norm.shape[0] * cam_norm.shape[1], cam_norm.shape[2]))
            cam_norm[cam_norm > 1] = 1
            cam_norm[cam_norm < 0] = 0
            label_cam[label_cam > 1] = 1
            label_cam[label_cam < 0] = 0

            # Write image
            img_no = datapacket_in['image_name'][cam_no]
            cv2.imwrite(os.path.join(cam_path, img_no) + '.png', cam_norm * 255)
            cv2.imwrite(os.path.join(label_cam_path, img_no) + '.png', label_cam * 255)
            print('Save cam : ', img_no, end='\r')
    print('')

def measure_cams(config: Config):
    config_json = config.toDictionary()
    print('measure_cams')
    print(config_json)
    import cv2
    import os
    import numpy as np
    from sklearn import metrics
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager
    from data.voc2012 import image_to_label

    # Set up data loader
    dataloader = DataLoader(
        Segmentation(
            config.classifier_dataset_root,
            source='train',
            augmentation='val',
            image_size=config.classifier_image_size,
            requested_labels=['classification', 'segmentation']
        ),
        batch_size=config.cams_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=4,
        prefetch_factor=4
    )

    wandb.init(entity='kobus_wits', project='wass_cams', name=config.sweep_id + '_cam_' + config.classifier_name, config=config_json)

    # Get cams directory
    cam_root_path = os.path.join(artifact_manager.getDir(), 'cam')

    count = 0

    for batch_no, batch in enumerate(dataloader):
        inputs_in = batch[0]
        labels_in = batch[1]
        datapacket_in = batch[2]

        if count > 1000:
            break

        for image_no, image_name in enumerate(datapacket_in['image_name']):
            image_width = datapacket_in['width'][image_no].numpy()
            image_height = datapacket_in['height'][image_no].numpy()
            channels = labels_in['classification'].shape[1]

            cam_path = os.path.join(cam_root_path, image_name + '.png')
            cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)

            cam_reshaped = np.reshape(cam, ((channels, image_height, image_width)))
            cam_reshaped = cam_reshaped / 255.0

            cam_background = np.power(1 - np.max(cam_reshaped, axis=0, keepdims=True), 4)
            cam_with_background = np.concatenate((cam_background, cam_reshaped), axis=0)

            cam_reshaped_max = np.argmax(cam_with_background, 0)
            for i in range(0, cam_with_background.shape[0]):
                indicies = cam_reshaped_max == i
                cam_with_background[i, :] = 0
                cam_with_background[i, indicies] = 1

            image = cv2.imread(datapacket_in['image_path'][image_no])
            label = cv2.imread(datapacket_in['label_path'][image_no])
            label = image_to_label(label)

            predi = cam_with_background

            acc = metrics.accuracy_score(np.argmax(label, 0).flatten(), np.argmax(predi, 0).flatten())
            mapr = metrics.average_precision_score(label[1:].flatten(), predi[1:].flatten())
            miou = metrics.jaccard_score(np.argmax(label, 0).flatten(), np.argmax(predi, 0).flatten(), average='macro')

            wandb.log({
                'accuracy': acc,
                'mapr': mapr,
                'miou': miou,
            }, step=count)

            if count % 10 == 0 and count < 201:
                image = image.astype(np.float32)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predi = label_to_image(predi).astype(np.float32)
                predi = cv2.cvtColor(predi, cv2.COLOR_BGR2RGB)
                wandb.log({
                    'image': wandb.Image(image),
                    'predi': wandb.Image(predi)
                }, step=count)
            

            count += 1

            print(count)

    wandb.finish()