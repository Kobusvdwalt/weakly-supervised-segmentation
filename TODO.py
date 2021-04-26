
#TODO:
"""
Build an image plotter to show the weak output, strong output, label and input image
Build method comparison plotters
"""

# Experiment outline:
"""
First we train 3 fully supervised networks.
* We show the miou over epoch
* We show per class miou over epoch
* We gather 5 input images, their labels, the outputs
* We gather 5 ouputs over epoch to show improvement
"""

"""
Then we train our weak networks.
"""

# EPIC:
# * Write up a section with results on each of the methods and our own best effort approach. 

# TASKS:
# * Measure deeplab
# * Measure fcn
# * Measure unet

# * Implement GAIN

# To deal with the noiseyness of the high resolution mask
# we can try to perform a blur or a downsample and use that for the loss.
# Hopefully this means that we can have precision whilst keeping coherent masks.

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
srun -N6 -p batch -l /bin/hostname
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
# Step1: rsync -chavzP --stats pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/models/checkpoints/multiclass_up.pt ./ 
# Step2: scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/multiclass_up.pt ./


# Talking points :
# * Global Max Pooling has the same effect as adding a regulizer to constrain erase masks
# * Global Max Pooling is better suited to adverserial sem seg, because it doesn't have GAP's spacial make up assumption
# 
# * We have novelty in the approach due to
#   - GMP
#   - Max Along Output Features for a single erase mask
#   - Full scale UNET
# 
# * Our loss function is problematic since it doesn't capture exactly what we desire.
#   - We desire a pixel to pixel class mapping
#   - But our loss function is only concerned with maximally decresing classifier accuracy
#   - THESE ARE NOT THE SAME !!
# 
#   


# Talking points 2.0:
# * All other weakly supervised semantic segmentation use a classifier to produce low resolution segmentation maps
# * This is problematic due to the loss of detailed features (from max pooling all the way down)
# * Training a semantic segmentation network in an adeverserial fashion allows us to capatalize on the detailed features
# 
# Pixel to Pixel training augmentation
# * What if we augment the current setup with actual end to end, pixel to pixel training ?
# * Would this help guide our other loss functions to a more attractive point ?
# 
# Softmax
# * Why doesn't softmax seem to work ??? Logically it should be fine


"""
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
"""