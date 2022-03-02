def start():
    from training.train_seg import train_seg

    sweeps = [
        {
            'dataset_root': 'datasets/generated/voc',
            'model_name': 'unet',
            'epochs': 51,
            'batch_size': 24,
            'image_size': 320
        },
    ]

    for sweep_index, sweep in enumerate(sweeps):
        print(f'Sweep start {sweep_index}/{len(sweeps)}')

        train_seg(
            dataset_root=sweep['dataset_root'],
            model_name=sweep['model_name'],
            epochs=sweep['epochs'],
            batch_size=sweep['batch_size'],
            image_size=sweep['image_size'],
        )