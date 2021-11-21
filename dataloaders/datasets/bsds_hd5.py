import os
import torch
import h5py
import random
import numpy as np
import scipy.io
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image,ImageStat
from torch.utils import data
from torchvision import transforms

def get_subwindow(im, label, center_pos, original_sz, avg_chans):
    """
     img
     pos: center
     original_sz: crop patch size = 320
    """
    if isinstance(center_pos, float):
        center_pos = [center_pos, center_pos]
    sz = original_sz
    im_sz = im.shape ## H,W
    c = (original_sz+1) / 2 # 320/2 = 160

    context_xmin = round(center_pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(center_pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1

    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    ## for example, if context_ymin<0, now context_ymin = 0
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    # avg_chans = np.array(avg_chans).reshape(3,)
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im

        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    v,r,c= label.shape
    avg_chans = np.array([0]).reshape(1, )
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_label = np.zeros((v, r + top_pad + bottom_pad, c + left_pad + right_pad),
                            np.float32)  # 0 is better than 1 initialization
        te_label[:,top_pad:top_pad + r, left_pad:left_pad + c] = label

        if top_pad:
            te_label[:,0:top_pad, left_pad:left_pad + c] = avg_chans
        if bottom_pad:
            te_label[:,r + top_pad:, left_pad:left_pad + c] = avg_chans
        if left_pad:
            te_label[:,:, 0:left_pad] = avg_chans
        if right_pad:
            te_label[:,:, c + left_pad:] = avg_chans
        label_patch_original = te_label[:,int(context_ymin):int(context_ymax + 1),
                               int(context_xmin):int(context_xmax + 1)]
    else:
        label_patch_original = label[:,int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]
    return im_patch_original,label_patch_original

class Mydataset(data.Dataset):
    def __init__(self,root_path='....../Augmentation/', split='trainval',crop_size=513):
        self.split = split
        self.crop_size=crop_size
        if self.split == 'trainval':
            list_file = os.path.join(root_path,'trainval_pair.lst')
        else:
            list_file = os.path.join(root_path, 'test.lst')

        with open(list_file, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]

        if self.split == 'trainval':
            pairs = [line.split() for line in lines]
            self.images_path = [pair[0] for pair in pairs]
            self.edges_path = [pair[1] for pair in pairs]
        else:
            self.images_path = lines
            self.images_name = []
            for path in self.images_path:
                folder, filename = os.path.split(path)
                name, ext = os.path.splitext(filename)
                self.images_name.append(name)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
                             transforms.ToTensor(),
                             normalize])  ## pre-process of pre-trained model of pytorch resnet-50
        ## ToTensor() need input (H,W,3)


    def __len__(self):
        return len(self.images_path)


    def __getitem__(self, idx):
        if self.split == 'trainval':
            ## img data
            img = Image.open(os.path.join(self.images_path[idx])).convert('RGB') # 读取图像，转换为三维矩阵
            w, h = img.size
            img_center = np.array([h / 2, w / 2]).astype(np.int)
            img = np.array(img)  ## (H,W,3) uint8 RGB
            avg_chans = np.mean(img, axis=(0, 1))  ##(3,)

            ## label data
            edge_path = os.path.join(self.edges_path[idx])
            h = h5py.File(edge_path, 'r')
            edge = np.squeeze(h['label'][...])
            label = edge.astype(np.float32)

            ##random crop 320x320
            offset_x, offset_y = 0, 0
            offset = True
            if offset:
                offset_y = int(100 * (random.random() - 0.5))
                offset_x = int(100 * (random.random() - 0.5))
            img_center = [img_center[0] + offset_y, img_center[1] + offset_x]
            img_crop, label_crop = get_subwindow(img, label, img_center, self.crop_size, avg_chans)

            img_tensor = self.trans(img_crop)
            label = torch.from_numpy(label_crop).float()

            return img_tensor, label
        else:
            img = Image.open(os.path.join(self.images_path[idx])).convert('RGB')
            img_tensor = self.trans(img)
            return img_tensor

if __name__ == '__main__':
    train_dataset = Mydataset(root_path='/*******/Augmentation/', split='trainval', crop_size=320)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                   num_workers=4, pin_memory=True, drop_last=True)
    tbar = tqdm(train_loader)
    num_img_tr = len(train_loader)
    for i, (image, target) in enumerate(tbar):
        print(target.shape)







