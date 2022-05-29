# Login to AWS proxy
"""
cd C:\Users\Kobus\
ssh -p443 -i details.pem ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com
"""
# Connect to WITS cluster and keep alive
"""
ssh -o ServerAliveInterval=10 pjvanderwalt@146.141.21.100
"""

# Activate anaconda
"""
source anaconda3/bin/activate
conda activate pytorch
"""

# Running Commands Directly
"""
srun -N6 -p batch -l /bin/hostname iaCi7EeR
srun -N2 -p biggpu -l cat /proc/cpuinfo | grep model
srun -N4 -p ha -l /usr/bin/uptime

sbatch --job-name=vgg11 --partition=batch --wrap="python train_voc_vgg11.py" --output=vgg11.out
sbatch --job-name=vgg13 --partition=batch --wrap="python train_voc_vgg13.py" --output=vgg13.out
sbatch --job-name=vgg16 --partition=batch --wrap="python train_voc_vgg16.py" --output=vgg16.out

sbatch --job-name=vgg16u0 --partition=batch --wrap="python train_voc_vgg16_unfreeze_0.py" --output=vgg16u0.out
sbatch --job-name=vgg16u1 --partition=batch --wrap="python train_voc_vgg16_unfreeze_1.py" --output=vgg16u1.out
sbatch --job-name=vgg16u2 --partition=batch --wrap="python train_voc_vgg16_unfreeze_2.py" --output=vgg16u2.out
sbatch --job-name=vgg16u3 --partition=batch --wrap="python train_voc_vgg16_unfreeze_3.py" --output=vgg16u3.out
"""

# Train on biggpu
"""
sbatch --job-name=wsgan --partition=biggpu --wrap="python train_voc_unet_adverserial.py" --output=wsgan.out
sbatch --job-name=wsgan --partition=biggpu --wrap="python main.py" --output=wsgan.out
"""

# Copy from WITS cluster to machine
"""
scp -r pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/models/checkpoints/
"""
# Copy from WITS cluster to AWS proxy
"""
scp -r pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/models/checkpoints/ ~/
scp -r pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/training/output/ ~/
"""

# Copy from AWS proxy to machine
"""
scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/checkpoints /
scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/output /
"""

# CHPC interactive session
"""
qsub -I -P CSCI1340 -q serial -l select=1:ncpus=1:mpiprocs=1:nodetype=haswell_reg
"""

# Copy specific weight file
"""
rsync -chavzP --stats pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/models/checkpoints/multiclass_up.pt ./ 
scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/multiclass_up.pt ./
"""




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




"""
Talking points :
* Global Max Pooling has the same effect as adding a regulizer to constrain erase masks
* Global Max Pooling is better suited to adverserial sem seg, because it doesn't have GAP's spacial make up assumption

* We have novelty in the approach due to
  - GMP
  - Max Along Output Features for a single erase mask
  - Full scale UNET

* Our loss function is problematic since it doesn't capture exactly what we desire.
  - We desire a pixel to pixel class mapping
  - But our loss function is only concerned with maximally decresing classifier accuracy
  - THESE ARE NOT THE SAME !!

  

Talking points 2.0:
* All other weakly supervised semantic segmentation use a classifier to produce low resolution segmentation maps
* This is problematic due to the loss of detailed features (from max pooling all the way down)
* Training a semantic segmentation network in an adeverserial fashion allows us to capatalize on the detailed features

Pixel to Pixel training augmentation
* What if we augment the current setup with actual end to end, pixel to pixel training ?
* Would this help guide our other loss functions to a more attractive point ?

Softmax
* Why doesn't softmax seem to work ??? Logically it should be fine

As it currently stands the adversary can never fool the classifier since the "mask" only moves the pixels torwards grey,
there are still features (not visible to humuns) but descriminative in the image. We need to force information loss somehow.
Simply adding noise doesn't work since the gradients will be very noisy the network won't move the weight in the direction
that knows not to mask or to mask that particular patch of images. Perhaps we should to a very hard sigmoid step

Although since the VGG feature extractor is not trainable, this image grey might still cause information loss since 
the lower level filter weights can't move in the right direction to pick up the non visible features.

This idea of an discriminator makes a ton of sense. It acts as a strong "meta" regulizer.
Since our loss function is an approximation of our ideal loss function there are gaps that the network can exploit.
One example is the zebra pattern but there are other instances like bottom and side bars apprearing. This is clearly a shortcut
to take that meets the classification loss demands without pushing up the mask-regulizer loss.
I almost feel like a discriminator would be valuable in any case where weak supervision is applied. Or any generation task.
If only to fight against the network learning any shortcuts.

Should we apply our soft threshold before mask generation or before erasing ?
Not before classification since that destroys the BCE gradient.

We also need to implement a paramater scheduler for the weights on the losses
and the random prob sampling

The discriminator is definately working
During training the emergence of the zebra artifact could be observed but was quickly eliminated
Instead of appearing in the middle of the mask the pattern has shifted towards the edges of the masks
which is excelent as this means the network is reducing mask size from outwards in.
We might end up with undercoverege masks but this can be fixed with morph operations
Time To Train up and till this point was 70 epochs

A possible improvement is still to have a random erase selector, to incentivise mask independance.
For example if the "person" mask was not used to erase, the "bike" mask should still erase the bike part of the image.
This needs a bit more thinking...

"""

# PG Supervision
# PG Supervision
# 100%
# 10
# D3

# - Put together a bullet point story with points on what results we currently and currently need.
# - Close/Threshold/CRF & Use that as the GT for the discriminator
# - Train the DeepLab v3 on the weak labels and compare.
# - Replace the UNET with a DeepLab v3
# - Run fair experiments of us vs find-it-if-you-can (blocks/cam and friends)
# - Train final deeplab on a single epoch output vs training over multiple outputs(breathing masks)
# - Also compare all of this on the blobby masks. Make a precision vs recall argument.
# - Train final semantic segmentation on both shape/blob datasets.
