if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.abspath('../'))

    from torch.utils.data import DataLoader
    from semseg.data import PascalVOCSegmentation
    from semseg.pascal_helper import LabelToImage
    from semseg.pascal_helper import LabelToClasses 
    from semseg.pascal_helper import ClassesToWords
    from semseg.pascal_helper import ThresholdClasses
    from semseg.pascal_helper import AddBackgroundClass
    from semseg.pascal_helper import BuildCam
    from classification.models.vgg_gap import vgg
    from metrics.iou import iou

    import numpy as np
    import cv2
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pascal_val = PascalVOCSegmentation(source='val')
    data = DataLoader(pascal_val, batch_size=16, shuffle=False, num_workers=8)

    vgg.train()
    for images, labels in data:
        inputs = images.permute(0, 3, 1, 2)

        inputs = inputs.float()
        labels = labels.float()

        # Get final conv layer activation
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        vgg.vgg16.features[28].register_forward_hook(get_activation('last_conv'))

        # Predict
        outputs = vgg(inputs)

        # Convert to numpy
        images_np = images.numpy()
        labels_np = labels.numpy()
        outputs_np = outputs.data.numpy()
        weights_np = vgg.dense1.weight.data.numpy()
        activations_np = activation['last_conv'].numpy()

        for sample_index in range(0, activations_np.shape[0]):
            classification_label = LabelToClasses(labels_np[sample_index])
            classification_label_words = ClassesToWords(classification_label)
            classification_pred = ThresholdClasses(outputs_np[sample_index])
            classification_pred_words = ClassesToWords(AddBackgroundClass(classification_pred))
            
            print('------------------------')
            print('Label: ' + str(classification_label_words))
            print('Prediction: ' + str(classification_pred_words))

            activations = activations_np[sample_index]
            weights = weights_np
            
            segmentation_pred = np.zeros(labels_np[sample_index].shape)
            for class_index in range(0, len(classification_pred)-1):
                if (classification_pred[class_index] == 1):
                    cam = BuildCam(activations, weights, class_index)
                    cam_resized = cv2.resize(cam, (256, 256), interpolation=cv2.INTER_NEAREST)
                    segmentation_pred[class_index + 1] = cam_resized
            
            prediction = LabelToImage(segmentation_pred)
            label = LabelToImage(labels_np[sample_index])
            # Show image and CAM
            cv2.imshow('input', images_np[sample_index])
            cv2.imshow('pred', prediction)
            cv2.imshow('label', label)
            cv2.waitKey(0)
            # print(iou(out, labels_np[sample, semseg_class]))

            cv2.waitKey(1)
