if __name__ == '__main__':
    from artifacts.artifact_manager import artifact_manager

    from datasets import voc2012_mask

    from training import train_voc_vgg16_blob
    from training import train_voc_vggm_blob

    from training import train_voc_deeplab101
    from training import train_voc_unet
    from training import train_voc_weak_unet
    from training import train_voc_wass
    from training import train_voc_gain
    from training import train_voc_gain_unet

    from visualize import visualize_voc_vgg16
    from visualize import visualize_voc_unet
    from visualize import visualize_voc_weak_unet
    from visualize import visualize_voc_wass

    # voc2012_mask.generate()

    # artifact_manager.setArtifactContainer("vgg16_blob")
    # train_voc_vgg16_blob.start()

    artifact_manager.setArtifactContainer("vggm_blob")
    train_voc_vggm_blob.start()

    # Weak=VGG16 | Strong=Unet
    # artifact_manager.setArtifactContainer("weak_vgg16_strong_unet")
    # train_voc_vgg16.start()
    # plot_training.plot_erase()
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


# Thoughts:
# I have moved away from GMP and Channel-max to previous work methods
# Haven't seen a major improvement other than training stability

# * I want to move from trying to find THE "solution",
# to writing about everything we have tried. I believe we have a lot to write about.

# * Perhaps we don't get our "crisp" segmentation map because the classifier
# is using any sort of detailed map to infer the class. How would we test this ?

# A small ammount of supervision is easy to implement within this architecture

# I want to train the classifier on the erased ground truth maps and see how it performs


# Next steps is probably to write a paper on this and compare to past work


# * 

# TODO:
# * At what "blobby" point does it fail ?
# * Add "Blobify" layer after segmentor


# * Preprocessing/augmentation of the input images
#   * Gradient image
#   * Smoothing/blur


# Thoughts
# * We have three problems in the adverserial approach
# * The first problem is that the network learns from shape

# * The second problem is that we are using the classifier loss on training data instead of validation data,
# so the network might overfit and thus introduce noise for the segmentor loss. Does this explain why we have decent results early on that get worse ?

# * Object context is a vital part of classification, with a large enough and diverse enough dataset this shouldn't be a problem. But for a small
# dataset like VOC2012 the network doesn't see the objects of interest in enough different scenes. An obvious example is boats and water. But there are less obvious examples


# * We can probably just train a network on the masks straight,
#       this would allow us to prove that training from shape is possible,
#       without interference from our two other problems context and overfitting

# * Two time update rule

# Story recap:
# * We want to train a semseg network on image level labels
# * A sensible way to do this is through adverserial training
# * Adverserial training is a proxy for our true loss function, and thus we have artifacts like checkboard, zebra
# * We can address these artifacts with a third network which is a dicriminator acting as a "meta" regulizer
# * We still don't get our crisp tight segmentation masks !!!
# * The classifier is probably learning from shape, how do we prove this ?
# * We can erase the human labeled masks (covering all discriminative regions) and see if the classifier can still train

# * The classifier trains remarkably well !! - First sanity check pass
# * If we destroy shape information does the ability to train deteriorate ? - Not enough :( what could be happening...
# * The dataset is small, so it's probably overfitting on small very descriminative little features
# * Class labels can be infered from contextual surroundings, so if boats are on water then the erase mask will erase the water. Not what we want.


# Password: xcvbmfgdGVHVGG
# Username: kobus@146.141.21.89
# kobus@lamp.ms.wits.ac.za

# Thoughts
# I am concerned about using a pretrained network
# Does this still allow us to make the argument that the classifier can learn from shape ? Probably

# Trying to train from scratch has been difficult, the network doesn't really perform well.
# Perhaps we should try training from scratch in our Adverserial Model ? Moving away from pretrained networks might help..