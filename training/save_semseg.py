
import torch
import torch.nn.functional as F
from data.voc2012 import label_to_image
from models.get_model import get_model

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
            source='val',
            source_augmentation='val',
            image_size=image_size,
            requested_labels=['classification', 'segmentation']
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=4
    )

    # Clear and create destination directory
    semseg_path = os.path.join(artifact_manager.getDir(), 'semseg')
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

            # Normalize each cam map to between 0 and 1
            # semseg_max = np.max(semseg, (1, 2), keepdims=True)
            # semseg_norm = semseg / (semseg_max + 1e-5)
            semseg_norm = semseg

            semseg_as_label = label_to_image(semseg_norm)
            cv2.imshow('img', semseg_as_label)
            cv2.imshow('lab', label_to_image(labels_in['segmentation'][0].detach().cpu().numpy()))
            cv2.waitKey(1)


            # Write image
            img_no = datapacket_in['image_name'][semseg_no]
            cv2.imwrite(os.path.join(semseg_path, img_no) + '.png', semseg_as_label * 255)
            print('Save cam : ', img_no, end='\r')
    print('')

def measure_semseg(
    dataset_root,
    model_name,
    batch_size=8,
    image_size=256,
    use_gt_labels=False,
):
    print('Measure semseg : ', locals())
    # import shutil
    # import cv2
    # import os
    # import numpy as np
    # from torch.utils.data.dataloader import DataLoader
    # from data.loader_segmentation import Segmentation
    # from artifacts.artifact_manager import artifact_manager
    # from metrics.iou import class_iou
    # from data.voc2012 import image_to_label

    # # Set up data loader
    # dataloader = DataLoader(
    #     Segmentation(
    #         dataset_root,
    #         source='train',
    #         source_augmentation='val',
    #         image_size=image_size,
    #         requested_labels=['classification', 'segmentation']
    #     ),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     prefetch_factor=4
    # )

    # # Get cams directory
    # cam_root_path = os.path.join(artifact_manager.getDir(), 'cam')

    # # Get results output path
    # cam_result_path = os.path.join(artifact_manager.getDir(), 'cam_results.txt')

    # # Open results file
    # cam_result_file = open(cam_result_path, 'w')

    # class_iou_sum = None
    # class_iou_count = None

    # for batch_no, batch in enumerate(dataloader):
    #     inputs_in = batch[0]
    #     labels_in = batch[1]
    #     datapacket_in = batch[2]

    #     if batch_no == 0:
    #         class_iou_sum = np.zeros(labels_in['segmentation'].shape[1])
    #         class_iou_count = np.zeros(labels_in['segmentation'].shape[1]) + 1e-4

    #     for image_no, image_name in enumerate(datapacket_in['image_name']):
    #         image_width = datapacket_in['width'][image_no].numpy()
    #         image_height = datapacket_in['height'][image_no].numpy()
    #         channels = labels_in['classification'].shape[1]

    #         cam_path = os.path.join(cam_root_path, image_name + '.png')
    #         cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)

    #         cam_reshaped = np.reshape(cam, ((channels, image_height, image_width)))
    #         cam_reshaped = cam_reshaped / 255.0

    #         cam_background = np.power(1 - np.max(cam_reshaped, axis=0, keepdims=True), 4)
    #         cam_with_background = np.concatenate((cam_background, cam_reshaped), axis=0)

    #         cam_reshaped_max = np.argmax(cam_with_background, 0)
    #         for i in range(0, cam_with_background.shape[0]):
    #             indicies = cam_reshaped_max == i
    #             cam_with_background[i, :] = 0
    #             cam_with_background[i, indicies] = 1

    #         image = cv2.imread(datapacket_in['image_path'][image_no])
    #         label = cv2.imread(datapacket_in['label_path'][image_no])
    #         label = image_to_label(label)

    #         class_iou_result = class_iou(cam_with_background, label, 0)

    #         # Increment count
    #         gt_classes = labels_in['classification'][image_no].numpy()
    #         gt_classes[gt_classes > 0.5] = 1
    #         gt_classes[gt_classes <= 0.5] = 0
    #         class_iou_count += np.concatenate([[1], gt_classes], axis=0)

    #         # Increment iou
    #         class_iou_sum += class_iou_result
    #         class_mean = class_iou_sum / class_iou_count

    #         print('class mean : ', class_mean, ' mean w b : ', np.mean(class_mean))

    #         cam_result_file.writelines([f'{image_name}:{class_iou_result}'])


    #         cv2.imshow('image', image)
    #         cv2.imshow('cam', cam_with_background[0])
    #         cv2.imshow('label', label[0])
    #         cv2.waitKey(1)

    # cam_result_file.close()
