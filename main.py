if __name__ == '__main__':
    from artifacts import artifact_manager
    from training import train_voc_deeplab101
    from training import train_voc_vgg16
    from training import train_voc_unet
    from training import train_voc_weak_unet

    from visualize import visualize_voc_vgg16
    from visualize import visualize_voc_unet
    from visualize import visualize_voc_weak_unet

    from measure import measure

    # Weak=VGG16 | Strong=Unet
    artifact_manager.instance.setArtifactContainer("weak_vgg16_strong_unet")
    # train_voc_vgg16.train_voc_vgg16()
    # visualize_voc_vgg16.visualize_voc_vgg16()
    # measure.measure(artifact_manager.instance.getArtifactDir() + "voc_vgg16_visualization/", "voc_vgg16_measurements")
    # train_voc_weak_unet.train_voc_weak_unet(artifact_manager.instance.getArtifactDir() + "voc_vgg16_visualization/")
    visualize_voc_weak_unet.visualize_voc_weak_unet()


    # Weak=WSSS | Strong=Unet
    # Weak=GAIN | Strong=Unet
    # Weak=ADVL | Strong=Unet
    
    # Supervised Unet
    # artifact_manager.instance.setArtifactContainer("supervised_unet")
    # train_voc_unet.train_voc_unet()
    # visualize_voc_unet.visualize_voc_unet()

    # Supervised Deeplab101
    artifact_manager.instance.setArtifactContainer("supervised_deeplab101")

    # Supervised FCN
    artifact_manager.instance.setArtifactContainer("supervised_fcn")