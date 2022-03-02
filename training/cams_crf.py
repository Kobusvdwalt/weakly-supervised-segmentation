
def _crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    import numpy as np

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def _crf_with_alpha(image, cam, alpha):
    import numpy as np
    import cv2
    import torch

    # Normalize Cam
    cam_wo_bg = cam[1:]
    cam_wo_bg = cam_wo_bg / (np.max(cam_wo_bg, (1, 2), keepdims=True) + 1e-5)    

    # Select only true lables
    cam_dict = {}
    for i in range(cam_wo_bg.shape[0]):
        if np.sum(cam_wo_bg[i]) > 0.01:
            cam_dict[i] = cam_wo_bg[i]

    cam_wo_bg = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(cam_wo_bg, axis=0, keepdims=True), alpha)
    cam_w_bg = np.concatenate((bg_score, cam_wo_bg), axis=0)

    cv2.imshow('back', cam_w_bg[0])
    cv2.waitKey(10)

    # Compute CRF
    # image = np.copy(image, order='C')
    crf_output_cam = _crf_inference(image, cam_w_bg, labels=cam_w_bg.shape[0])

    # Build output cam
    cam_out = np.zeros(cam.shape)
    cam_out[0] = crf_output_cam[0]
    for cam_no, cam_key in enumerate(cam_dict):
        cam_out[cam_key+1] = crf_output_cam[cam_no+1]

    return cam_out

def save_crf(
    dataset_root,
    image_size=256,
    low_alpha=4,
    high_alpha=32,
):
    print('Apply CRF on cams : ', locals())
    import shutil
    import cv2
    import os
    import numpy as np
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from data.voc2012 import label_to_image
    from artifacts.artifact_manager import artifact_manager
    from data.voc2012 import get_augmentation_crf

    cam_root_path = os.path.join(artifact_manager.getDir(), 'cam')

    # Get augmentation
    augmentation = get_augmentation_crf(image_size)

    # Set up data loader
    dataloader = DataLoader(
        Segmentation(
            dataset_root,
            source='trainval',
            image_size=image_size
        ),
        batch_size=16,
        shuffle=False ,
        num_workers=6,
        pin_memory=True
    )

    # Create high, low dirs
    cam_la_path = os.path.join(artifact_manager.getDir(), 'cam_la')
    if (os.path.exists(cam_la_path)):
        shutil.rmtree(cam_la_path)
    os.makedirs(cam_la_path)

    cam_ha_path = os.path.join(artifact_manager.getDir(), 'cam_ha')
    if (os.path.exists(cam_ha_path)):
        shutil.rmtree(cam_ha_path)
    os.makedirs(cam_ha_path)

    for batch_no, batch in enumerate(dataloader):
        inputs_in = batch[0]
        labels_in = batch[1]
        datapacket_in = batch[2]

        for image_no, image_name in enumerate(datapacket_in['image_name']):
            print(image_name)

            cam_path = os.path.join(cam_root_path, image_name + '.png')
            cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)

            shape_sum = cam.shape[0] * cam.shape[1]
            channels = (int)(shape_sum / (image_size * image_size))
            cam_reshaped = np.reshape(cam, ((channels, image_size, image_size)))
            cam_reshaped = cam_reshaped / 255.0

            image = cv2.imread(datapacket_in['image_path'][image_no])
            image = augmentation(image=image)['image']

            cv2.imshow('image', image)

            cam_la = _crf_with_alpha(image, cam_reshaped, low_alpha)
            cam_la = cam_la.reshape(cam_la.shape[0] * cam_la.shape[1], cam_la.shape[1])
            cv2.imwrite(os.path.join(cam_la_path, image_name + '.png'), cam_la * 255)

            cam_ha = _crf_with_alpha(image, cam_reshaped, high_alpha)
            cam_ha = cam_ha.reshape(cam_ha.shape[0] * cam_ha.shape[1], cam_ha.shape[1])
            cv2.imwrite(os.path.join(cam_ha_path, image_name + '.png'), cam_ha * 255)

            

            # cam_ha = _crf_with_alpha(cam_reshaped, high_alpha)

            # cam_la = cam_la.reshape(cam_la.shape[0] * cam_la.shape[1], cam_la.shape[1])
            # cam_ha = cam_la.reshape(cam_ha.shape[0] * cam_ha.shape[1], cam_ha.shape[1])
            
            
            # cv2.imwrite(os.path.join(cam_ha_path, image_name + '.png'), cam_ha * 255)
            
            # cv2.imshow('cam_as_image', cam_as_image)
            # cv2.imshow('label_as_image', label_as_image)
            # cv2.imshow('campath', cam)
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
