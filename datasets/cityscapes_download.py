import gdown
url = 'https://drive.google.com/uc?id=1wn3uofULi1eoTgSGRw9_b4YlzXEe17cP&export=download'
output = 'cityscapes.zip'
gdown.download(url, output, quiet=False)