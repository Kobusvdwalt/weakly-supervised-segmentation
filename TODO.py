
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