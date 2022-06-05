import cv2
import wandb
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from data.voc2012 import image_to_label, label_to_image
from models.get_model import get_model
from tools.average_meter import AverageMeter
from training.config_manager import Config

def save_cams(config: Config):
    config_json = config.toDictionary()
    print('save_cams')
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
    model.eval()
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
        batch_size=config.cams_produce_batch_size,
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

            cam_bg = (1 - np.max(cam_norm, axis=0, keepdims=True)) ** config.cams_bg_alpha
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


def _measure_sample(payload):
    count = payload['count']
    image_path = payload['image_path']
    cam_path = payload['cam_path']
    label_path = payload['label_path']
    predi_path = payload['predi_path']
    
    label = cv2.imread(label_path)
    predi = cv2.imread(predi_path)

    label = image_to_label(label)
    predi = image_to_label(predi)

    accuracy = metrics.accuracy_score(np.argmax(label, 0).flatten(), np.argmax(predi, 0).flatten())
    mapr = metrics.average_precision_score(label[1:].flatten(), predi[1:].flatten())
    miou = metrics.jaccard_score(np.argmax(label, 0).flatten(), np.argmax(predi, 0).flatten(), average='macro')

    log = {
        'accuracy': accuracy,
        'mapr': mapr,
        'miou': miou,
        'count': count
    }

    if count < 8:
        raw = cv2.imread(cam_path)
        image = cv2.imread(image_path)
        image = image.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predi = label_to_image(predi).astype(np.float32)
        predi = cv2.cvtColor(predi, cv2.COLOR_BGR2RGB)
        log['raw_' + str(count)] = wandb.Image(raw)
        log['img_' + str(count)] = wandb.Image(image)
        log['pred_' + str(count)] = wandb.Image(predi)
        log['count'] = 0

    return log


def measure_cams(config: Config):
    config_json = config.toDictionary()
    print('measure_cams')
    print(config_json)
    import os
    
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager
    from multiprocessing import Pool

    # Set up data loader
    dataloader = DataLoader(
        Segmentation(
            config.eval_dataset_root,
            source='train',
            augmentation='val',
            image_size=config.classifier_image_size,
            requested_labels=['classification', 'segmentation']
        ),
        batch_size=config.cams_measure_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=2,
        prefetch_factor=2
    )

    # Get cams directory
    cam_root_path = os.path.join(artifact_manager.getDir(), 'cam')
    label_cam_path = os.path.join(artifact_manager.getDir(), 'labels_cam')

    count = 0
    
    wandb.init(entity='kobus_wits', project='wass_measure_cams', name=config.sweep_id + '_cam_' + config.classifier_name, config=config_json)
    avg_meter = AverageMeter('accuracy', 'mapr', 'miou')

    for batch_no, batch in enumerate(dataloader):
        datapacket = batch[2]

        payloads = []
        for image_no, image_name in enumerate(datapacket['image_name']):
            payload = {
                'count': count,
                'image_path': datapacket['image_path'][image_no],
                'label_path': datapacket['label_path'][image_no],
                'predi_path': os.path.join(label_cam_path, image_name + '.png'),
                'cam_path': os.path.join(cam_root_path, image_name + '.png'),
            }
            payloads.append(payload)
            count += 1
            print('Measure cam : ', count, end='\r')

        with Pool(8) as poel:
            logs = poel.map(_measure_sample, payloads)

            for log in logs:
                avg_meter.add({
                    'accuracy': log['accuracy'],
                    'mapr': log['mapr'],
                    'miou': log['miou'],
                })

                if log['count'] < 8:
                    wandb.log(log, step=log['count'])

            wandb.log({
                'accuracy': avg_meter.get('accuracy'),
                'mapr': avg_meter.get('mapr'),
                'miou': avg_meter.get('miou'),
            })

    wandb.finish()
