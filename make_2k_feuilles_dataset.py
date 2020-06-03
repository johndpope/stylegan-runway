import os
import joblib
import shutil

import numpy as np

from glob import glob
from PIL import Image
from tqdm import tqdm

base_path = '/media/romain/windows/Users/Romain/Downloads/archives2k/'

toreverse = ['chene vert', 'laurier rose']

dirnames = os.listdir(base_path)
filelist = []
counter = 0
for d in dirnames:
    img_list = glob(base_path + d + '/*.png')
    n = len(img_list)
    res = None
    for img_path in img_list:
        img = Image.open(img_path)
        if res is None:
            res = img.size
            print(img.size)
        elif img.size != res:
            print(img.size, img_path)

        tokens = d.split('_')
        tree = ' '.join(tokens[3:]).lower()
        if tree in toreverse:
            img.rotate(180).save(img_path)

    filelist += img_list
    print(n, d)


## Copy from base to data path and fliplr
data_path = '/media/romain/windows/Users/Romain/Desktop/stylegan2/dataset/feuilles2k/'
base_path = '/media/romain/data/Projects/Datasets/Feuilles/ToAdd/'
img_list = glob(base_path + '*/*/*.png')
#img_list = glob(data_path + 'feuilles2k/*.png')

for img_path in tqdm(img_list):
    fname = os.path.basename(img_path)
    img = Image.open(img_path)
    mirrored = Image.fromarray(np.fliplr(np.array(img)))
    tokens = fname.split('_')
    flip_fname = '_'.join(tokens[:-1]) + '_1' + tokens[-1][1:]
    mirrored.save(os.path.join(data_path, flip_fname))
    img.save(os.path.join(data_path, fname))

# copy every subdirectory
dirnames = os.listdir(base_path)
filelist = []
counter = 0
for d in dirnames:
    img_list = glob(base_path + d + '/*.png')
    for i, img_path in enumerate(img_list):
        dest_path = os.path.join(data_path, 'feuilles2k', d + '_%04d.png' %i)
        print(shutil.copy(img_path, dest_path))


# reformat glycine names
rename_path = base_path + '/Glycine/'
img_list = glob(rename_path + '*/*.png')
for i, img_path in enumerate(img_list):
    fname = os.path.basename(img_path)
    dest_path = os.path.dirname(img_path)
    tokens = fname.split('_')
    order = [1, 2, 0, 3]
    new_fname = '_'.join([tokens[k] for k in order])
    shutil.move(img_path, dest_path + '/' + new_fname)

# rename tieule
for fpath in filelist:
    fname = os.path.basename(fpath)
    # tokens = os.path.dirname(fname).split('_')
    _path = os.path.dirname(fpath)
    tokens = fname.split('_')
    if tokens[0].lower() == 'tieule':
        name, res, res1, side, num = tokens
        rename = '_'.join([side, res, res1, name, num])
        shutil.move(fpath, _path + '/' + rename)

from sklearn.preprocessing import LabelEncoder
# get file name list
side, tree = [], []
for fname in filelist:
    # tokens = os.path.dirname(fname).split('_')
    tokens = os.path.basename(fname).split('_')
    side.append(tokens[0].lower())
    tree.append(' '.join(tokens[1:-1]).lower())

side_enc = LabelEncoder()
side_enc.fit(side)

joblib.dump(side_enc, 'side_label_encoder.pkl')

tree_enc = LabelEncoder()
tree_enc.fit(tree)

joblib.dump(tree_enc, 'tree_label_encoder.pkl')
