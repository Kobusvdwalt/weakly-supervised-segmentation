def start():
    from time import time
    from training.train_classifier import train_classifier
    from training.train_affinitynet import train_affinitynet

    from training.save_cams import save_cams
    from training.save_cams import measure_cams
    from training.save_cams_crf import save_cams_crf
    from training.save_cams_random_walk import save_cams_random_walk
    from training.save_cams_random_walk import measure_random_walk
    from training.train_semseg import train_semseg
    from training.save_semseg import save_semseg
    from training.save_semseg import measure_semseg
    from artifacts.artifact_manager import artifact_manager
    from training.config_manager import Config
    from training.config_manager import config_manager

    sweeps = [
        # Config(
        #     eval_dataset_root='datasets/generated/voc',
        #     classifier_dataset_root='datasets/generated/voc_aug',
        #     classifier_name='vgg16',
        #     classifier_epochs=1,
        #     classifier_batch_size_train=32,
        #     classifier_pretrained=True,
        #     classifier_pretrained_unfreeze=10,
            
        #     cams_produce_batch_size=32,
        #     cams_measure_batch_size=64,

        #     affinity_net_batch_size=8,

        #     semseg_dataset_root='datasets/generated/voc_aug'
        # ),

        Config(
            eval_dataset_root='datasets/generated/voc',
            classifier_dataset_root='datasets/generated/voc_aug',
            classifier_name='wasscam',
            classifier_epochs=100,
            classifier_batch_size_train=8,
            classifier_pretrained=True,
            classifier_pretrained_unfreeze=10,
            
            cams_produce_batch_size=32,
            cams_measure_batch_size=64,

            affinity_net_batch_size=8,

            semseg_dataset_root='datasets/generated/voc_aug'
        ),
    ]

    for sweep_index, sweep in enumerate(sweeps):
        artifact_manager.setArtifactContainer('psa_sweep_' + str(sweep_index))
        sweep.sweep_id = str(time())
        config_manager.setConfig(sweep)
        print(f'Sweep start {sweep_index}/{len(sweeps)}')

        ########## CAMS
        # Train the classifier
        train_classifier(sweep)

        # # Save out the CAMs
        # save_cams(sweep)

        # # Measure cams
        # measure_cams(sweep)

        # ########## AFFINITY NET
        # # Apply DCRF on CAMs
        # save_cams_crf(sweep)

        # # Train AffinityNet
        # train_affinitynet(sweep)

        # # Perform random walk
        # save_cams_random_walk(sweep)

        # # Measure random walk
        # measure_random_walk(sweep)
        
        # # ########## SEM SEG
        # # Train segmentation network
        # train_semseg(sweep)

        # # Save segmentation outputs
        # save_semseg(sweep)

        # # Measure segmentation outputs
        # measure_semseg(sweep)
