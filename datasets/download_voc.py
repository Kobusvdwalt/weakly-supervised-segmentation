import gdown
import zipfile
url = 'https://drive.google.com/uc?id=1wn3uofULi1eoTgSGRw9_b4YlzXEe17cP&export=download'
output = 'voc2012.zip'
gdown.download(url, output, quiet=False)

with zipfile.ZipFile('voc2012.zip', 'r') as zip_ref:
    zip_ref.extractall('voc2012')