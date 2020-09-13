import zipfile
with zipfile.ZipFile('cityscapes.zip', 'r') as zip_ref:
    zip_ref.extractall()