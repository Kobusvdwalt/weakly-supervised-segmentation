# Cityscapes has to be split into two files to stay within google drive's 10gb public file
# limit. cityscapes1.zip contains the input images and GT labels for the train split.
# cityscapes2.zip contains the images and labels for the val and test splits.
# Each of these zip files has a sub folder called cityscapes so they can be extracted as is.
# The result is a single folder with all splits merged.

import gdown
url = 'https://drive.google.com/uc?id=1xCccVI3T-N78HGL9YmQtMG07kvVKvHbj&export=download'
output = 'cityscapes1.zip'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1xCccVI3T-N78HGL9YmQtMG07kvVKvHbj&export=download'
output = 'cityscapes2.zip'
gdown.download(url, output, quiet=False)
