
import torch
import torch.nn.functional as F
from data.voc2012 import label_to_image
from models.get_model import get_model
from training.config_manager import Config

def save_cams_random_walk(config: Config):
    config_json = config.toDictionary()
    print('save_cams_random_walk')
    print(config_json)
    import shutil
    import cv2
    import os
    import numpy as np
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager

    # Set up model
    model = get_model(config.affinity_net_name)
    model.load()
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
        num_workers=4,
        prefetch_factor=4
    )

    # Get cam source directory
    cam_path = os.path.join(artifact_manager.getDir(), 'cam')

    # Clear and create output directory
    labels_rw_path = os.path.join(artifact_manager.getDir(), 'labels_rw')
    if (os.path.exists(labels_rw_path)):
        shutil.rmtree(labels_rw_path)
    os.makedirs(labels_rw_path)

    for batch_no, batch in enumerate(dataloader):
        inputs_in = batch[0]
        labels_in = batch[1]
        datapacket_in = batch[2]

        image = inputs_in['image'].cuda(non_blocking=True)

        for image_no, image_name in enumerate(datapacket_in['image_name']):
            print('image name ', image_name)
            image_width = datapacket_in['width'][image_no].numpy()
            image_height = datapacket_in['height'][image_no].numpy()
            channels = labels_in['classification'].shape[1]
            
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
            cam_background = np.power(1 - np.max(cam, axis=0, keepdims=True), 4)
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
                aff_mat = torch.pow(aff_mat, 8)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(8):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_pooled = F.avg_pool2d(torch.from_numpy(cam_padded), 8, 8)

            # cam_vis = cam_pooled[0].clone().cpu().numpy()

            # def get_aff_sum(aff_in):
            #     aff_lab = np.sum(aff_in, axis=0) / aff_in.shape[0]
            #     aff_lab = np.reshape(aff_lab, (cam_vis.shape[0], cam_vis.shape[1]))
            #     return aff_lab

            # aff_vis = get_aff_sum(trans_mat.clone().cpu().numpy())
            # aff_vis = cv2.resize(aff_vis, (aff_vis.shape[1] * 8, aff_vis.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
            # aff_vis_normed = aff_vis - np.min(aff_vis)
            # aff_vis_normed = aff_vis_normed / (np.max(aff_vis_normed) + 1e-5)
            # cv2.imshow('aff_vis', aff_vis * 256)
            # cv2.imshow('aff_vis_normed', aff_vis_normed)            
            # cam_vis = cv2.resize(cam_vis, (cam_vis.shape[1] * 8, cam_vis.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow('cam_bg', cam_vis)
            # cv2.waitKey(0)

            cam_vec = cam_pooled.view(21, -1)

            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, image_height_pooled, image_width_pooled)

            cam_rw = torch.nn.Upsample((image_height_padded, image_width_padded), mode='bilinear')(cam_rw)
            cam_rw = cam_rw.cpu().data[0, :, :image_height, :image_width]

            label_rw = label_to_image(cam_rw)

            cv2.imwrite(os.path.join(labels_rw_path, image_name + '.png'), label_rw * 255)