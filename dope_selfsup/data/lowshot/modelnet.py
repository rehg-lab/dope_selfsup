import os.path as osp
import os
import numpy as np
import cv2
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pprint import pprint
from torchvision.transforms import functional as TF

from dope_selfsup.data import data_utils

IMAGE_PATH = './dataset_directory/modelnet_lowshot/images'  
SPLIT_PATH = './dataset_directory/modelnet_lowshot' 

class ModelNet(Dataset):

    def __init__(self, setname, split_dir="split_000"):
        csv_path = osp.join(SPLIT_PATH, split_dir, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []
        
        obj_set = []
        
        curr_obj = ''
        for i, l in enumerate(lines):
            name, wnid = l.split(',')
            
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            obj = ''.join(name.split('_')[:-1])
            
            if i == 0: 
                curr_obj = obj
                obj_ls = []
            
            if obj != curr_obj:
                data.append(obj_ls)
                label.append(lb)
                curr_obj = obj
                obj_ls = []

            obj_ls.append(path)
        
        self.data = data  # data path of all data
        self.labels = label  # label of all data
        self.num_class = len(set(label))
        print("="*50) 
        print("Split:", setname)
        print("Total objects:", len(self.data))
        print("Categories:", self.wnids)
        print("="*50) 
        
        self.setname=setname
        augmentation_file = "augmentation_settings.json" 

        if setname == "train":
            if augmentation_file is not None and augmentation_file != "none":
                with open(f"./dope_selfsup/data/{augmentation_file}", "r") as f:
                    aug_params = json.load(f)
                pprint(aug_params)
                self.transform = data_utils.ContrastiveAugmentation(aug_params)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        paths, label = self.data[i], self.labels[i]
        path = np.random.choice(paths)
        
        img = Image.open(path).convert('RGB')
        
        mask_path = path.replace('images', 'masks')
        
        seg = data_utils.read_segmentation(mask_path)
        seg = np.stack([seg, seg, seg], axis=-1).astype(np.uint8)
        seg = Image.fromarray(seg)
        
        uv_c1 = np.ones((10,2)) ## so we can reuse the augmentation logic from the contrastive models
        if self.setname == "train":
            bool1 = np.random.choice([True, False])
            img, seg, _ = self.transform(img, seg, uv_c1, bool1)

        img = TF.to_tensor(img)

        return img, label

    
if __name__ == '__main__':
    SEED = 1234
    import matplotlib.pyplot as plt

    ## viz dataloading  
    out_dir = "dataset_test_outputs/ModelNet_lowshot_dataset_output"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset= ModelNet("train", "split_000")
    for i in np.random.randint(0,len(dataset), 40):
        print(i)
        image, label = dataset.__getitem__(i)
        
        fig, ax = plt.subplots(1)

        image = image.numpy().transpose(1,2,0)
        ax.imshow(image)
        ax.axis("off")
        
        fig.tight_layout()
        fig.savefig(
                os.path.join(out_dir, "{:04d}.png".format(i))
            )


