
import wandb
import cv2
import os
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from training.config_manager import Config

def _crf_inference(img, probs, t=10, scale_factor=1, labels=21):
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
    # Normalize Cam
    cam_wo_bg = cam[1:]
    cam_wo_bg = cam_wo_bg / (np.max(cam_wo_bg, (1, 2), keepdims=True) + 1e-5)

    # Select only true lables
    cam_dict = {}
    for i in range(cam.shape[0]):
        if np.sum(cam[i]) > 0.01:
            cam_dict[i] = cam[i]

    cam_true_labels = np.array(list(cam_dict.values()))

    if cam_true_labels.shape[0] == 0:
        return np.zeros(cam.shape)

    cam_background = (1 - np.max(cam_true_labels, axis=0, keepdims=True)) ** alpha
    cam_with_background = np.concatenate((cam_background, cam_true_labels), axis=0)

    # Compute CRF
    crf_output_cam = _crf_inference(image, cam_with_background, labels=cam_with_background.shape[0])

    # Build output cam
    cam_out = np.zeros((cam.shape[0] + 1, cam.shape[1], cam.shape[2]))
    cam_out[0] = crf_output_cam[0]
    for cam_no, cam_key in enumerate(cam_dict):
        cam_out[cam_key+1] = crf_output_cam[cam_no+1]

    return cam_out

def _process_sample(payload):
    image_name = payload['image_name']
    image_path = payload['image_path']
    image_width = payload['image_width']
    image_height = payload['image_height']
    channels = payload['channels']
    count = payload['count']
    cam_root_path = payload['cam_root_path']
    cam_la_path = payload['cam_la_path']
    cam_ha_path = payload['cam_ha_path']
    alpha_high = payload['alpha_high']
    alpha_low = payload['alpha_low']

    cam_path = os.path.join(cam_root_path, image_name + '.png')
    cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)

    cam_reshaped = np.reshape(cam, ((channels, image_height, image_width)))
    cam_reshaped = cam_reshaped / 255.0

    image = cv2.imread(image_path)

    cam_la = _crf_with_alpha(image, cam_reshaped, alpha_low)
    cam_la = cam_la.reshape(cam_la.shape[0] * cam_la.shape[1], cam_la.shape[2])
    cv2.imwrite(os.path.join(cam_la_path, image_name + '.png'), cam_la * 255)

    cam_ha = _crf_with_alpha(image, cam_reshaped, alpha_high)
    cam_ha = cam_ha.reshape(cam_ha.shape[0] * cam_ha.shape[1], cam_ha.shape[2])
    cv2.imwrite(os.path.join(cam_ha_path, image_name + '.png'), cam_ha * 255)

    if count < 8:
        return {
            'crf_la_' + str(count): wandb.Image(cam_la),
            'crf_ha_' + str(count): wandb.Image(cam_ha),
            'image_count': 0,
        }
    else:
        return {'image_count': count}

def save_cams_crf(config: Config):
    config_json = config.toDictionary()
    print('save_cams_crf')
    print(config_json)
    import shutil
    import numpy as np
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from data.voc2012 import label_to_image
    from multiprocessing import Pool
    from artifacts.artifact_manager import artifact_manager

    cam_root_path = os.path.join(artifact_manager.getDir(), 'cam')

    # Set up data loader
    dataloader = DataLoader(
        Segmentation(
            config.classifier_dataset_root,
            source='train',
            image_size=config.classifier_image_size
        ),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2
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

    wandb.init(entity='kobus_wits', project='wass_measure_cams_crfs', name=config.sweep_id + '_cam_' + config.classifier_name, config=config_json)
    count = 0

    for batch_no, batch in enumerate(dataloader):
        from training.save_cams_crf import _process_sample
        
        labels = batch[1]
        datapacket = batch[2]

        payloads = []
        for image_no, image_name in enumerate(datapacket['image_name']):
            payload = {
                'image_name': image_name,
                'count': count,
                'image_width': datapacket['width'][image_no].numpy(),
                'image_height': datapacket['height'][image_no].numpy(),
                'channels': labels['classification'].shape[1],
                'image_path': datapacket['image_path'][image_no],
                'cam_la_path': cam_la_path,
                'cam_ha_path': cam_ha_path,
                'cam_root_path': cam_root_path,
                'alpha_low': config.cams_bg_alpha_low,
                'alpha_high': config.cams_bg_alpha_high,
            }
            payloads.append(payload)
            count += 1
            print('Save cam : ', count, end='\r')

        with Pool(8) as poel:
            logs = poel.map(_process_sample, payloads)

            for log in logs:
                wandb.log(log, step=log['image_count'])

    print('')
    wandb.finish()