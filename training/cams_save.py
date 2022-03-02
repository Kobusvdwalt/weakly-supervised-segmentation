
from shutil import Error


def get_model(model_name):
    model = None
    if model_name == 'vgg16':
        from models.vgg16 import Vgg16GAP
        model = Vgg16GAP(name="vgg16")
        return model

    if model_name == 'unet':
        from models.unet import UNet
        model = UNet()
        return model

    if model_name == 'deeplab':
        from models.deeplab import DeepLab
        model = DeepLab()
        return model

    raise Error('Model name has no implementation')

def save_cams(
    dataset_root,
    model_name,
    batch_size=8,
    image_size=256,
):
    print('Save cams : ', locals())
    import shutil
    import cv2
    import os
    import numpy as np
    from torch.utils.data.dataloader import DataLoader
    from data.loader_segmentation import Segmentation
    from data.voc2012 import label_to_image
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
            image_size=image_size
        ),
        batch_size=batch_size,
        shuffle=False ,
        num_workers=6,
        pin_memory=True
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
        cams = model.event({
            'name': 'get_cam',
            'inputs': inputs_in,
            'labels': labels_in,
            'batch': batch_no+1
        })

        # Save out cams
        for cam_no, cam in enumerate(cams):
            # Save out labels
            cam = labels_in['segmentation'][cam_no].numpy()

            for i in range(0, cam.shape[0]):
                cam[i] = cv2.blur(cam[i], (32, 32))

            # Disregard false positives
            gt_mask = labels_in['classification'][cam_no].numpy()
            gt_mask[gt_mask > 0.5] = 1
            gt_mask[gt_mask <= 0.5] = 0
            gt_mask = np.expand_dims(np.expand_dims(gt_mask, -1), -1)
            cam[1:, :, :] *= gt_mask

            # Set background to 0.5, will be computed later
            cam[0] = 0.5
            img_no = datapacket_in['image_name'][cam_no]
            cam = cam.reshape(cam.shape[0] * cam.shape[1], cam.shape[1])
            print(img_no)
            cv2.imwrite(os.path.join(cam_path, img_no) + '.png', cam * 255)

        cv2.imshow('image', label_to_image(cams[0]))
        cv2.waitKey(1)
