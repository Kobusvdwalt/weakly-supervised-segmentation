
def start():
    from training.train_model import train_model
    from training.cams_save import save_cams
    from training.cams_crf import save_crf

    from artifacts.artifact_manager import artifact_manager

    sweeps = [
        {
            'classifier_dataset_root': 'datasets/generated/voc',
            'classifier_name': 'vgg16',
            'classifier_epochs': 20,
            'classifier_batch_size': 16,
            'classifier_image_size': 360
        },
    ]

    for sweep_index, sweep in enumerate(sweeps):
        artifact_manager.setArtifactContainer('psa_sweep_' + str(sweep_index))
        print(f'Sweep start {sweep_index}/{len(sweeps)}')

        # Train the classifier
        # train_model(
        #     dataset_root=sweep['classifier_dataset_root'],
        #     model_name=sweep['classifier_name'],
        #     epochs=sweep['classifier_epochs'],
        #     batch_size=sweep['classifier_batch_size'],
        #     image_size=sweep['classifier_image_size'],
        # )

        # Save out the CAMs
        # save_cams(
        #     dataset_root=sweep['classifier_dataset_root'],
        #     model_name=sweep['classifier_name'],
        #     batch_size=sweep['classifier_batch_size'],
        #     image_size=sweep['classifier_image_size'],
        # )

        # Apply DCRF on CAMs
        save_crf(
            dataset_root=sweep['classifier_dataset_root'],
            image_size=sweep['classifier_image_size'],
        )


        # Train AffinityNet
        
        # Perform random walk

        # Train segmentation network
        