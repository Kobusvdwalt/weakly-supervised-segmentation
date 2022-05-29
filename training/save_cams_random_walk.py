


import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from data.voc2012 import image_to_label, label_to_image
from models.get_model import get_model
from tools.average_meter import AverageMeter
from training.config_manager import Config
from sklearn import metrics

def save_cams_random_walk(config: Config):
    config_json = config.toDictionary()
    print('save_cams_random_walk')
    print(config_json)
    import shutil
    import os
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager

    # Set up model
    model = get_model(config.affinity_net_name)
    model.load()
    model.eval()
    model.to(model.device)

    # Set up data loader
    dataloader = DataLoader(
        Segmentation(
            config.classifier_dataset_root,
            source='train',
            augmentation='affinity_predict',
            image_size=config.affinity_net_image_size,
            requested_labels=['classification', 'segmentation']
        ),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2
    )

    # Get cam source directory
    cam_path = os.path.join(artifact_manager.getDir(), 'cam')

    # Clear and create output directory
    labels_rw_path = os.path.join(artifact_manager.getDir(), 'labels_rw')
    if (os.path.exists(labels_rw_path)):
        shutil.rmtree(labels_rw_path)
    os.makedirs(labels_rw_path)

    count = 0

    for batch_no, batch in enumerate(dataloader):
        inputs = batch[0]
        labels = batch[1]
        datapacket = batch[2]

        for image_no, image_name in enumerate(datapacket['image_name']):
            image = inputs['image'].cuda(non_blocking=True)
            image_width = datapacket['width'][image_no].numpy()
            image_height = datapacket['height'][image_no].numpy()
            channels = labels['classification'].shape[1]
            
            # Pad image
            image_width_padded = int(np.ceil(image_width/8)*8)
            image_height_padded = int(np.ceil(image_height/8)*8)
            image_padded = F.pad(image, (0, image_width_padded - image_width, 0, image_height_padded - image_height))

            image_width_pooled = int(np.ceil(image_width_padded/8))
            image_height_pooled= int(np.ceil(image_height_padded/8))

            # Load cam
            cam_path_instance = os.path.join(cam_path, image_name + '.png')
            cam = cv2.imread(cam_path_instance, cv2.IMREAD_GRAYSCALE)
            cam = np.reshape(cam, ((channels, image_height, image_width)))
            cam = cam / 255.0

            # Build cam background
            cam_background = (1 - np.max(cam, (0), keepdims=True))**config.affinity_net_bg_alpha
            cam = np.concatenate((cam_background, cam), axis=0)
            cam = cam.astype(np.float32)

            # Pad cam
            cam_padded_width = int(np.ceil(cam.shape[2]/8)*8)
            cam_padded_height = int(np.ceil(cam.shape[1]/8)*8)
            cam_padded = np.pad(cam, ((0, 0), (0, cam_padded_height - image_height), (0, cam_padded_width - image_width)), mode='constant')

            # Run images through model and get affinity matrix
            with torch.no_grad():
                aff_mat = model.event({
                    'name': 'infer_aff_net_dense',
                    'image': image_padded,
                })
                aff_mat = torch.pow(aff_mat, config.affinity_net_beta)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(config.affinity_net_log_t):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_pooled = F.avg_pool2d(torch.from_numpy(cam_padded), 8, 8)

            cam_vec = cam_pooled.view(21, -1)

            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, image_height_pooled, image_width_pooled)

            cam_rw = torch.nn.Upsample((image_height_padded, image_width_padded), mode='bilinear')(cam_rw)
            cam_rw = cam_rw.cpu().data[0, :, :image_height, :image_width]

            label_rw = label_to_image(cam_rw)

            cv2.imwrite(os.path.join(labels_rw_path, image_name + '.png'), label_rw * 255)

            count += 1
            print('Save cam : ', count, end='\r')

    print('')







def _measure_sample(payload):
    count = payload['count']
    label_path = payload['label_path']
    predi_path = payload['predi_path']

    label = cv2.imread(label_path)
    predi = cv2.imread(predi_path)

    predi = cv2.resize(predi, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)

    label = image_to_label(label)
    predi = image_to_label(predi)

    accuracy = metrics.accuracy_score(np.argmax(label, 0).flatten(), np.argmax(predi, 0).flatten())
    mapr = metrics.average_precision_score(label[1:].flatten(), predi[1:].flatten())
    miou = metrics.jaccard_score(np.argmax(label, 0).flatten(), np.argmax(predi, 0).flatten(), average='macro')

    return {
        'accuracy': accuracy,
        'mapr': mapr,
        'miou': miou,
        'count': count
    }

def measure_random_walk(config: Config):
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
        batch_size=config.affinity_net_batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2
    )

    # Get cams directory
    labels_rw_root_path = os.path.join(artifact_manager.getDir(), 'labels_rw')

    count = 0
    
    wandb.init(entity='kobus_wits', project='wass_measure_cams_rw', name=config.sweep_id + '_cam_' + config.classifier_name, config=config_json)
    avg_meter = AverageMeter('accuracy', 'mapr', 'miou')

    for batch_no, batch in enumerate(dataloader):
        datapacket_in = batch[2]

        payloads = []
        logs = []
        for image_no, image_name in enumerate(datapacket_in['image_name']):
            payload = {
                'count': count,
                'label_path': datapacket_in['label_path'][image_no],
                'predi_path': os.path.join(labels_rw_root_path, image_name + '.png'),
            }
            payloads.append(payload)
            logs.append(_measure_sample(payload))
            count += 1
            print('Measure cam RW : ', count, end='\r')

        # with Pool(8) as poel:
        # logs = poel.map(_measure_sample, payloads)

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
