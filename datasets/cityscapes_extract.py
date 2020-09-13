import zipfile
with zipfile.ZipFile('cityscapes1.zip', 'r') as zip_ref:
    zip_ref.extractall()

with zipfile.ZipFile('cityscapes2.zip', 'r') as zip_ref:
    zip_ref.extractall()