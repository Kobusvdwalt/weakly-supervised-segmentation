

# class PSASweepInput:

def start():
    from training.train_classifier import train_classifier
    from training.train_affinitynet import train_affinitynet

    from training.save_cams import save_cams
    from training.save_cams import measure_cams
    from training.save_cams_crf import save_cams_crf
    from training.save_cams_random_walk import save_cams_random_walk
    from training.train_semseg import train_semseg
    from training.save_semseg import save_semseg
    from training.save_semseg import measure_semseg
    from artifacts.artifact_manager import artifact_manager
    from training.config_manager import Config
    from training.config_manager import config_manager

    sweeps = [
        # Config(
        #     classifier_dataset_root='datasets/generated/voc_aug',
        #     classifier_name='vgg16',
        #     classifier_epochs=10,
        #     classifier_batch_size_train=8,
        #     classifier_batch_size_cams=16,
        #     affinity_net_batch_size=8,
        #     classifier_pretrained=False,
        #     classifier_pretrained_unfreeze=-1
        # ),
        Config(
            semseg_dataset_root='datasets/generated/voc_aug',
            semseg_name='deeplab',
            semseg_pretrained=True,
            semseg_batch_size=4,
            semseg_epochs=1
        ),
    ]

    for sweep_index, sweep in enumerate(sweeps):
        artifact_manager.setArtifactContainer('psa_sweep_' + str(sweep_index))
        config_manager.setConfig(sweep)
        print(f'Sweep start {sweep_index}/{len(sweeps)}')

        # # Train the classifier
        # train_classifier(sweep)

        # Train segmentation network
        train_semseg(sweep)

        # # Save out the CAMs
        # save_cams(
        #     dataset_root=sweep['classifier_dataset_root'],
        #     model_name=sweep['classifier_name'],
        #     batch_size=sweep['classifier_batch_size_cams'],
        #     image_size=sweep['classifier_image_size'],
        #     use_gt_labels=sweep['cams_save_gt_labels'],
        # )

        # # Measure cams
        # measure_cams(
        #     dataset_root=sweep['classifier_dataset_root'],
        #     model_name=sweep['classifier_name'],
        #     batch_size=sweep['classifier_batch_size_cams'],
        #     image_size=sweep['classifier_image_size'],
        #     use_gt_labels=sweep['cams_save_gt_labels'],
        # )

        # # Apply DCRF on CAMs
        # save_cams_crf(
        #     dataset_root=sweep['classifier_dataset_root'],
        #     image_size=sweep['classifier_image_size'],
        # )

        # # Train AffinityNet
        # train_affinitynet(
        #     dataset_root=sweep['classifier_dataset_root'],
        #     image_size=sweep['classifier_image_size'],
        #     batch_size=sweep['affinity_net_batch_size']
        # )
        
        # # Perform random walk
        # save_cams_random_walk(
        #     dataset_root=sweep['classifier_dataset_root'],
        #     image_size=sweep['classifier_image_size'],
        #     batch_size=sweep['affinity_net_batch_size']
        # )

        # # Transform cams into a functional dataset


        #################################################################
        #### SEGMENTATION ###############################################
        #################################################################



        # # Save segmentation outputs
        # save_semseg(
        #     dataset_root='datasets/generated/voc',
        #     model_name=sweep['semseg_name'],
        #     image_size=sweep['classifier_image_size'],
        #     batch_size=sweep['semseg_batch_size'],
        # )

        # # Measure segmentation outputs
        # measure_semseg(
        #     dataset_root='datasets/generated/voc',
        # )
