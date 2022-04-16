
import torch
import torch.nn.functional as F
from models.get_model import get_model

def save_labels(
    dataset_root,
    model_name,
    batch_size=8,
    image_size=256,
    use_gt_labels=False,
):
    print('Save cams : ', locals())
    import shutil
    import cv2
    import os
    import numpy as np
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager

    # Set up model
    model = get_model(model_name)
    model.load()
    model.to(model.device)

    # Set up data loader
    dataloader = DataLoader(
        Segmentation(
            dataset_root,
            source='trainval',
            source_augmentation='val',
            image_size=image_size,
            requested_labels=['classification', 'segmentation']
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Clear and create desintation directory
    cam_path = os.path.join(artifact_manager.getDir(), 'cam')
    if (os.path.exists(cam_path)):
        shutil.rmtree(cam_path)
    os.makedirs(cam_path)

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
            if use_gt_labels:
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

            # Upsample CAM to original image size
            # - Calculate original image aspect ratio
            width = datapacket_in['width'][cam_no].detach().numpy()
            height = datapacket_in['height'][cam_no].detach().numpy()
            aspect_ratio = width / height

            # - Calculate width and height to cut from upscaled CAM
            if aspect_ratio > 1:
                cut_width = image_size
                cut_height = round(image_size / aspect_ratio)
            else:
                cut_width = round(image_size * aspect_ratio)
                cut_height = image_size

            # - Upscale CAM to match input size
            cam = np.moveaxis(cam, 0, -1)
            cam = cv2.resize(cam, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            cam = np.moveaxis(cam, -1, 0)

            # - Cut CAM from input size and upscale to original image size 
            cam = cam[:, 0:cut_height, 0:cut_width]
            cam = np.moveaxis(cam, 0, -1)
            cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
            cam = np.moveaxis(cam, -1, 0)

            # Normalize each cam map to between 0 and 1
            cam_max = np.max(cam, (1, 2), keepdims=True)
            cam_norm = cam / (cam_max + 1e-5)

            # Collapse cam from 3d into long 2d
            cam_norm = np.reshape(cam_norm, (cam_norm.shape[0] * cam_norm.shape[1], cam_norm.shape[2]))
            cam_norm[cam_norm > 1] = 1
            cam_norm[cam_norm < 0] = 0

            # Write image
            img_no = datapacket_in['image_name'][cam_no]
            cv2.imwrite(os.path.join(cam_path, img_no) + '.png', cam_norm * 255)
            print('Save cam : ', img_no, end='\r')
    print('')