
def save_semseg(
    dataset_root,
    model_name,
    batch_size=8,
    image_size=256,
    use_gt_labels=False,
):
    print('Save semseg : ', locals())
    import shutil
    import cv2
    import os
    import numpy as np
    import torch.nn.functional as F
    import torch
    from models.get_model import get_model
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from artifacts.artifact_manager import artifact_manager
    from data.voc2012 import label_to_image
    
    # Set up model
    model = get_model(model_name)
    model.load()
    model.to(model.device)
    model.train(False)

    # Set up data loader
    dataloader = DataLoader(
        Segmentation(
            dataset_root,
            source='val',
            source_augmentation='val',
            image_size=image_size,
            requested_labels=['classification', 'segmentation']
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=4,
    )

    # Clear and create destination directory
    semseg_path = os.path.join(artifact_manager.getDir(), 'semseg_output')
    if (os.path.exists(semseg_path)):
        shutil.rmtree(semseg_path)
    os.makedirs(semseg_path)

    for batch_no, batch in enumerate(dataloader):
        inputs_in = batch[0]
        labels_in = batch[1]
        datapacket_in = batch[2]

        # Run images through model and get raw cams
        with torch.no_grad():
            semsegs = model.event({
                'name': 'get_semseg',
                'inputs': inputs_in,
                'labels': labels_in,
                'batch': batch_no+1
            })

            semsegs = semsegs.detach().cpu( ).numpy()

        # Save out cams
        for semseg_no, semseg in enumerate(semsegs):
            # Save out ground truth labels for testing the rest of the system
            if use_gt_labels:
                semseg = labels_in['segmentation'][semseg_no][1:]
                semseg = F.adaptive_avg_pool2d(semseg, [32, 32]).numpy()

                for i in range(0, semseg.shape[0]):
                    semseg[i] = cv2.blur(semseg[i], (3, 3))
                    semseg[i] = cv2.blur(semseg[i], (3, 3))

            # # Disregard false positives
            # gt_mask = labels_in['classification'][semseg_no].numpy()
            # gt_mask[gt_mask > 0.5] = 1
            # gt_mask[gt_mask <= 0.5] = 0
            # gt_mask = np.expand_dims(np.expand_dims(gt_mask, -1), -1)
            # cam *= gt_mask

            # Upsample CAM to original image size
            # - Calculate original image aspect ratio
            width = datapacket_in['width'][semseg_no].detach().numpy()
            height = datapacket_in['height'][semseg_no].detach().numpy()
            aspect_ratio = width / height

            # - Calculate width and height to cut from upscaled CAM
            if aspect_ratio > 1:
                cut_width = image_size
                cut_height = round(image_size / aspect_ratio)
            else:
                cut_width = round(image_size * aspect_ratio)
                cut_height = image_size

            # - Upscale CAM to match input size
            semseg = np.moveaxis(semseg, 0, -1)
            semseg = cv2.resize(semseg, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            semseg = np.moveaxis(semseg, -1, 0)

            # - Cut CAM from input size and upscale to original image size 
            semseg = semseg[:, 0:cut_height, 0:cut_width]
            semseg = np.moveaxis(semseg, 0, -1)
            semseg = cv2.resize(semseg, (width, height), interpolation=cv2.INTER_LINEAR)
            semseg = np.moveaxis(semseg, -1, 0)

            semseg_as_label = label_to_image(semseg)

            # Write image
            img_no = datapacket_in['image_name'][semseg_no]
            cv2.imwrite(os.path.join(semseg_path, img_no) + '.png', semseg_as_label * 255)
            print('Save cam : ', img_no, end='\r')
    print('')

def measure_semseg():
    print('awe')
