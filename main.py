if __name__ == '__main__':
    from artifacts.artifact_manager import artifact_manager
    from training import train_voc_deeplab101
    from training import train_voc_vgg16
    from training import train_voc_unet
    from training import train_voc_weak_unet
    from training import train_voc_wass
    from training import train_voc_gain
    from training import train_voc_gain_unet

    from visualize import visualize_voc_vgg16
    from visualize import visualize_voc_unet
    from visualize import visualize_voc_weak_unet
    from visualize import visualize_voc_wass

    from measure import measure
    from measure import plot_training
    from measure import plot_images

    # Weak=VGG16 | Strong=Unet
    artifact_manager.setArtifactContainer("weak_vgg16_strong_unet")
    # train_voc_vgg16.start()
    plot_training.plot_erase()
    # plot_images.plt_images()
    # train_voc_vgg16.train_voc_vgg16()
    # visualize_voc_vgg16.visualize_voc_vgg16()
    # plot_training.plot_training(artifact_manager.getDir() + "voc_vgg16_training_log.json")
    # measure.measure("voc_vgg16_visualization/", "voc_vgg16_measurements")
    # train_voc_weak_unet.train_voc_weak_unet(artifact_manager.getDir() + "voc_vgg16_visualization/")
    # visualize_voc_weak_unet.visualize_voc_weak_unet()

    # artifact_manager.setArtifactContainer("weak_wass_strong_unet")
    # train_voc_gain.start()
    # train_voc_gain_unet.start()
    # train_voc_wass.start()


    # Weak=WSSS | Strong=Unet
    # artifact_manager.setArtifactContainer("weak_wass_strong_unet")
    # train_voc_wass.train_voc_wass()
    # visualize_voc_wass.visualize_voc_wass()
    # train_voc_weak_unet.train_voc_weak_unet(artifact_manager.getDir() + "voc_wass_visualization/")


    # Weak=GAIN | Strong=Unet
    # Weak=ADVL | Strong=Unet
    
    # Supervised Unet
    # artifact_manager.setArtifactContainer("supervised_unet")
    # plot_training.plot_training(artifact_manager.getDir() + "voc_unet_training_log.json")
    # plot_images.plot_images()
    # train_voc_unet.train_voc_unet()
    # plot_training.plot_training(artifact_manager.getDir() + "voc_unet_training_log.json")
    # visualize_voc_unet.visualize_voc_unet()
    # measure.measure("voc_unet_visualization/", "voc_unet_measurements")

    # Supervised Deeplab101
    # artifact_manager.setArtifactContainer("supervised_deeplab101")

    # Supervised FCN
    # artifact_manager.setArtifactContainer("supervised_fcn")