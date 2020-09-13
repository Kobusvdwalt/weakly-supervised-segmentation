import zipfile
with zipfile.ZipFile('voc2012.zip', 'r') as zip_ref:
    zip_ref.extractall()