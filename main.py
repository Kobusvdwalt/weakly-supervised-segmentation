if __name__ == '__main__':
    from artifacts.artifact_manager import artifact_manager
    from experiments import exp_erase

    exp_erase.start()

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

# TOTALLY RANDOM THOUGHT ALERT !!!
# COCO has this image with a cat, people and planes. Totally random and unrealistic.
# This is great for highlighting the segmentation boundries of an object an detatching an object from it's context.. contradictory to the dataset's name :)
# Still, this idea is interesting, could we use segmentation masks as a data augmentation technique. Currently I am thinking of small modifications,
# like shifting a person 50 pixels left or right in an image. But can we go more extreme ?
# First we grey out all semantic context in our image,
# then we have a bunch of background images,
# next we cut (based on the object mask) images out and randomly paste them on our background images.
# 
# I'm thinking now, do we even need background images ? if we have a sufficently large supply of semanticly meaningful image cuts,
# can we not then produce an infinite amount of semantic segmentation training samples.. How much of an effect does surrounding context really have ?
# TOTALLY RANDOM THOUGHT IS NOW OVER !!!

# I would love to add a social studies/human experiment here where we see up till what level humans can use blobified images to find discriminative images.
# Then further more, what impact does it have when you have seen segmentation masks vs having not seen them before ?
# What does this mean for researcher bias in this experiment ?


# Found a paper that claims "Deep convolutional networks do not classify based on global object shape"
# * Seems like a low-quality effort, but it has an interesting idea...
# we can use our segmentation "mask" to generate RGB images that pass through random RGB images, the only discriminative feature would be shape. 
# This would be an even stronger proof of our assumtion that a classifier can learn from shape. Didn't spend too much time reading the paper but will do so soon
# Even if this fails, which the paper seems to suggest, I think it raises interesting questions as to why the mask only classifier would succeed as we've shown
# but the mask pass through fails... After some more thinkinf I would say this is probably due to the noise the RGB sections of the image introduce  


# I think we can confidently state that classifiers CAN learn from shape.
# The VOCO expirement did exactly what was expected in that it closed the train/val f1 score gap
# The overfitting point can be shown cleanly with a ever increasing dataset size. Whilst showing the train/val gap AND the baseline/erased gap
# My worry is still that the network is able to learn from shape so well. Even in the cases where we destroy shape a lot.
# The only thing left that can explain this behaviour is the contextual information. Object surrounds etc
# We can reasonably infer that because our mask-only experiment shows a more significant drop off in f1 score that context plays a part