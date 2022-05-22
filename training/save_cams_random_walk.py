
import torch
import torch.nn.functional as F
from models.get_model import get_model

def save_cams_random_walk(
    dataset_root,
    model_name = 'affinitynet',
    batch_size=8,
    image_size=256,
    use_gt_labels=False,
):
    print('Save cams random walk: ', locals())
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
            source='train',
            source_augmentation='val',
            image_size=image_size,
            requested_labels=['classification', 'segmentation']
        ),
        batch_size=batch_size,
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

        # Run images through model and get affinity matrix
        with torch.no_grad():
            aff_mat = model.event({
                'name': 'infer_aff_net_dense',
                'inputs': inputs_in,
            })

        for image_no, image_name in enumerate(datapacket_in['image_name']):
            cam_path_instance = os.path.join(cam_path, image_name + '.png')
            cam = cv2.imread(cam_path_instance, cv2.IMREAD_GRAYSCALE)

            image_width = datapacket_in['image_width'][image_no].detach().numpy()
            image_height = datapacket_in['image_width'][image_no].detach().numpy()
            # content_width = datapacket_in['content_width'][image_no].detach().numpy()
            # content_height = datapacket_in['content_height'][image_no].detach().numpy()

            
            dheight = int(np.ceil(image_height/8))
            dwidth = int(np.ceil(image_width/8))


            print('aff_mat.shape', aff_mat.shape)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)

            print('trans_mat.shape', trans_mat.shape)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)
            print('cam_full_arr.shape', cam_full_arr.shape)

            cam_vec = cam_full_arr.view(21, -1)
            
            print('cam_vec.shape', cam_vec.shape)

            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)

            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
            _, cam_rw_pred = torch.max(cam_rw, 1)

            res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]

            imageio.imsave(os.path.join(args.out_rw, name + '.png'), res)
            

        # # Save out cams
        # for cam_no, cam in enumerate(cams):
        #     # Save out ground truth labels for testing the rest of the system
        #     if use_gt_labels:
        #         cam = labels_in['segmentation'][cam_no][1:]
        #         cam = F.adaptive_avg_pool2d(cam, [32, 32]).numpy()

        #         for i in range(0, cam.shape[0]):
        #             cam[i] = cv2.blur(cam[i], (3, 3))
        #             cam[i] = cv2.blur(cam[i], (3, 3))

        #     # Disregard false positives
        #     gt_mask = labels_in['classification'][cam_no].numpy()
        #     gt_mask[gt_mask > 0.5] = 1
        #     gt_mask[gt_mask <= 0.5] = 0
        #     gt_mask = np.expand_dims(np.expand_dims(gt_mask, -1), -1)
        #     cam *= gt_mask

        #     # Upsample CAM to original image size
        #     # - Calculate original image aspect ratio
        #     width = datapacket_in['width'][cam_no].detach().numpy()
        #     height = datapacket_in['height'][cam_no].detach().numpy()
        #     aspect_ratio = width / height

        #     # - Calculate width and height to cut from upscaled CAM
        #     if aspect_ratio > 1:
        #         cut_width = image_size
        #         cut_height = round(image_size / aspect_ratio)
        #     else:
        #         cut_width = round(image_size * aspect_ratio)
        #         cut_height = image_size

        #     # - Upscale CAM to match input size
        #     cam = np.moveaxis(cam, 0, -1)
        #     cam = cv2.resize(cam, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        #     cam = np.moveaxis(cam, -1, 0)

        #     # - Cut CAM from input size and upscale to original image size 
        #     cam = cam[:, 0:cut_height, 0:cut_width]
        #     cam = np.moveaxis(cam, 0, -1)
        #     cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
        #     cam = np.moveaxis(cam, -1, 0)

        #     # Normalize each cam map to between 0 and 1
        #     cam_max = np.max(cam, (1, 2), keepdims=True)
        #     cam_norm = cam / (cam_max + 1e-5)

        #     # Collapse cam from 3d into long 2d
        #     cam_norm = np.reshape(cam_norm, (cam_norm.shape[0] * cam_norm.shape[1], cam_norm.shape[2]))
        #     cam_norm[cam_norm > 1] = 1
        #     cam_norm[cam_norm < 0] = 0

        #     # Write image
        #     img_no = datapacket_in['image_name'][cam_no]
        #     cv2.imwrite(os.path.join(cam_path, img_no) + '.png', cam_norm * 255)
        #     print('Save cam : ', img_no, end='\r')
    print('')

# def measure_cams(
#     dataset_root,
#     model_name,
#     batch_size=8,
#     image_size=256,
#     use_gt_labels=False,
# ):
#     print('Measure cams : ', locals())
#     import shutil
#     import cv2
#     import os
#     import numpy as np
#     from torch.utils.data.dataloader import DataLoader
#     from data.loader_segmentation import Segmentation
#     from artifacts.artifact_manager import artifact_manager
#     from metrics.iou import class_iou
#     from data.voc2012 import image_to_label

#     # Set up data loader
#     dataloader = DataLoader(
#         Segmentation(
#             dataset_root,
#             source='train',
#             source_augmentation='val',
#             image_size=image_size,
#             requested_labels=['classification', 'segmentation']
#         ),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         prefetch_factor=4
#     )

#     # Get cams directory
#     cam_root_path = os.path.join(artifact_manager.getDir(), 'cam')

#     # Get results output path
#     cam_result_path = os.path.join(artifact_manager.getDir(), 'cam_results.txt')

#     # Open results file
#     cam_result_file = open(cam_result_path, 'w')

#     class_iou_sum = None
#     class_iou_count = None

#     for batch_no, batch in enumerate(dataloader):
#         inputs_in = batch[0]
#         labels_in = batch[1]
#         datapacket_in = batch[2]

#         if batch_no == 0:
#             class_iou_sum = np.zeros(labels_in['segmentation'].shape[1])
#             class_iou_count = np.zeros(labels_in['segmentation'].shape[1]) + 1e-4

#         for image_no, image_name in enumerate(datapacket_in['image_name']):
#             image_width = datapacket_in['width'][image_no].numpy()
#             image_height = datapacket_in['height'][image_no].numpy()
#             channels = labels_in['classification'].shape[1]

#             cam_path = os.path.join(cam_root_path, image_name + '.png')
#             cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)

#             cam_reshaped = np.reshape(cam, ((channels, image_height, image_width)))
#             cam_reshaped = cam_reshaped / 255.0

#             cam_background = np.power(1 - np.max(cam_reshaped, axis=0, keepdims=True), 4)
#             cam_with_background = np.concatenate((cam_background, cam_reshaped), axis=0)

#             cam_reshaped_max = np.argmax(cam_with_background, 0)
#             for i in range(0, cam_with_background.shape[0]):
#                 indicies = cam_reshaped_max == i
#                 cam_with_background[i, :] = 0
#                 cam_with_background[i, indicies] = 1

#             image = cv2.imread(datapacket_in['image_path'][image_no])
#             label = cv2.imread(datapacket_in['label_path'][image_no])
#             label = image_to_label(label)

#             class_iou_result = class_iou(cam_with_background, label, 0)

#             # Increment count
#             gt_classes = labels_in['classification'][image_no].numpy()
#             gt_classes[gt_classes > 0.5] = 1
#             gt_classes[gt_classes <= 0.5] = 0
#             class_iou_count += np.concatenate([[1], gt_classes], axis=0)

#             # Increment iou
#             class_iou_sum += class_iou_result
#             class_mean = class_iou_sum / class_iou_count

#             print('class mean : ', class_mean, ' mean w b : ', np.mean(class_mean))

#             cam_result_file.writelines([f'{image_name}:{class_iou_result}'])


#             cv2.imshow('image', image)
#             cv2.imshow('cam', cam_with_background[0])
#             cv2.imshow('label', label[0])
#             cv2.waitKey(1)

#     cam_result_file.close()
