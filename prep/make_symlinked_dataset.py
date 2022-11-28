import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--src_p", type=str, default="")
parser.add_argument("--dest_p", type=str, default="")
args = parser.parse_args()

src_p = args.src_p 
dest_p = args.dest_p

dest_p_im = os.path.join(dest_p, "images")
dest_p_mask = os.path.join(dest_p, "masks")   

if not os.path.exists(dest_p_im):
    os.makedirs(dest_p_im)

if not os.path.exists(dest_p_mask):
    os.makedirs(dest_p_mask)

objects = [x for x in os.listdir(src_p) if '.obj' in x]

paths = []

for obj in objects:

    img_dirpath = os.path.join(src_p, obj, "segmentations", obj.replace('.obj', ''))
    images = os.listdir(img_dirpath)

    for im in images:
        assert os.path.exists(os.path.join(img_dirpath, im))
        paths.append(os.path.join(img_dirpath, im))

for src_impath in paths:
    im_num = src_impath.split('/')[-1].replace('.jpeg','').split('_')[-1]
    obj = src_impath.split('/')[-4].replace('.obj', '')
    
    dest_impath = os.path.join(dest_p_mask, f"{obj}_{im_num}.jpeg")
    os.symlink(src_impath, dest_impath)

paths = []

for obj in objects:

    img_dirpath = os.path.join(src_p, obj, "RGB")
    images = os.listdir(img_dirpath)

    for im in images:
        assert os.path.exists(os.path.join(img_dirpath, im))
        paths.append(os.path.join(img_dirpath, im))

for src_impath in paths:
    im_num = src_impath.split('/')[-1].replace('.jpeg', '')
    obj = src_impath.split('/')[-3].replace('.obj', '')
    
    dest_impath = os.path.join(dest_p_im, f"{obj}_{im_num}.jpeg") 
    os.symlink(src_impath, dest_impath)
