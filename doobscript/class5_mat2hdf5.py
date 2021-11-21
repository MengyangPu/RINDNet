# --------------------------------------------------------
# Copyright (c) 2018 Guoxia Wang
# DOOBNet data augmentation and converting tool
# Written by Guoxia Wang
# --------------------------------------------------------

import sys
import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
from PIL import ImageFilter
import scipy.ndimage
import argparse


plt.rcParams['figure.figsize'] = 20, 5

val_cnt = 0
train_cnt = 0
total_pixel = 0
total_edge_pixel = 0


def gt_flip(edge_ori):
    bg = edge_ori[0, 0, ...]
    depth = edge_ori[0, 1, ...]
    normal = edge_ori[0, 2, ...]
    reflectance = edge_ori[0, 3, ...]
    illumination = edge_ori[0, 4, ...]

    bg = np.fliplr(bg)
    depth = np.fliplr(depth)
    normal = np.fliplr(normal)
    reflectance = np.fliplr(reflectance)
    illumination = np.fliplr(illumination)

    edge_ori[0, 0, ...] = bg
    edge_ori[0, 1, ...] = depth
    edge_ori[0, 2, ...] = normal
    edge_ori[0, 3, ...] = reflectance
    edge_ori[0, 4, ...] = illumination

    return edge_ori

def gt_rotate(edge_ori, times):
    bg = edge_ori[0, 0, ...]
    depth = edge_ori[0, 1, ...]
    normal = edge_ori[0, 2, ...]
    reflectance = edge_ori[0, 3, ...]
    illumination = edge_ori[0, 4, ...]

    bg = np.rot90(bg, times)
    depth = np.rot90(depth, times)
    normal = np.rot90(normal, times)
    reflectance = np.rot90(reflectance, times)
    illumination = np.rot90(illumination, times)

    height, width = depth.shape
    edge_ori = np.zeros((1, 5, height, width), dtype=np.float32)

    edge_ori[0, 0, ...] = bg
    edge_ori[0, 1, ...] = depth
    edge_ori[0, 2, ...] = normal
    edge_ori[0, 3, ...] = reflectance
    edge_ori[0, 4, ...] = illumination

    return edge_ori


def BSDS__augmentation(mat_path, h5_dir, img_src_dir, img_dst_dir):
    mat_id = os.path.splitext(os.path.split(mat_path)[1])[0]
    mat = scipy.io.loadmat(mat_path)
    gt = mat['gtStruct']['gt_theta'][0][0][0]

    bg = gt[0][:, :, 0]
    depth = gt[0][:, :, 1]
    normal = gt[0][:, :, 2]
    reflectance = gt[0][:, :, 3]
    illumination = gt[0][:, :, 4]
    height, width = depth.shape
    label = np.zeros((1, 5, height, width), dtype=np.float32)
    label[0, 0, ...] = bg
    label[0, 1, ...] = depth
    label[0, 2, ...] = normal
    label[0, 3, ...] = reflectance
    label[0, 4, ...] = illumination
    img_src_filename = os.path.join(img_src_dir, '{}.jpg'.format(mat_id))
    img = Image.open(img_src_filename)

    for rot in range(4):
        img_rot = img
        label_rot = label
        if (rot != 0):
            img_rot = img.transpose(Image.ROTATE_90 + rot - 1)  # Image.ROTATE_90 = 2
            label_rot = gt_rotate(label.copy(), rot)

        for flip in range(2):
            img_rot_flip = img_rot
            label_rot_flip = label_rot
            if (flip > 0):
                img_rot_flip = img_rot.transpose(Image.FLIP_LEFT_RIGHT)
                label_rot_flip = gt_flip(label_rot.copy())

            filename = '{}_rot{}_flip{}'.format(mat_id, rot * 90, flip)
            # print filename
            img_dst_filename = os.path.join(img_dst_dir, '{}.jpg'.format(filename))
            img_rot_flip.save(img_dst_filename, quality=100)
            h5_filename = os.path.join(h5_dir, '{}.h5'.format(filename))
            with h5py.File(h5_filename, 'w') as f:
                f['label'] = label_rot_flip

def parse_args():
    parser = argparse.ArgumentParser(description='BSDS-RIND dataset converting and augmenting tool')
    parser.add_argument(
        '--dataset', default='BSDS-RIND', help="dataset name, BSDS-RIND or BSDSownership",
        type=str)
    parser.add_argument(
        '--label-dir', default='*****/BSDS-RIND/mat',help="the label directory that contains *.mat label files",
        type=str)
    parser.add_argument(
        '--img-dir', default='******/BSDS-RIND/images/trainval', help="the source image directory",
        type=str)
    parser.add_argument(
        '--test-img-dir', default='*****/BSDS-RIND/images/test',help="the source image directory",
        type=str)
    parser.add_argument(
        '--output-dir', default='******/BSDS-RIND/Augmentation', help="the directory that save to augmenting images and .h5 label files",
        type=str)
    parser.add_argument(
        '--bsdsownership-testfg', default='******/BSDS-RIND/testgt/depth', help="testfg directory, it only require for BSDS ownership dataset",
        type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    dst_label_dir = os.path.join(args.output_dir, 'Aug_HDF5EdgeOriLabel')
    dst_img_dir = os.path.join(args.output_dir, 'Aug_JPEGImages')
    if (not os.path.exists(dst_label_dir)):
        os.mkdir(dst_label_dir)
    if (not os.path.exists(dst_img_dir)):
        os.mkdir(dst_img_dir)

    if (args.dataset == 'BSDSownership'):
        print('Start converting and augmenting {} dataset ...'.format(args.dataset))

        mat_list = glob.glob(os.path.join(args.label_dir, '*.mat'))
        for mat_path in mat_list:
            BSDS__augmentation(mat_path, dst_label_dir, args.img_dir, dst_img_dir)    

        # generate train and test file list for training and testing and test ids for matlab evaluation code
        train_val_pair_list = []
        test_list = []
        test_iids = []

        h5_list = glob.glob(os.path.join(dst_label_dir, '*.h5'))
        for h5_path in h5_list:
            h5_filename = os.path.split(h5_path)[1]
            h5_id = os.path.splitext(h5_filename)[0]

            # here use abspath such that you can move train and test list file to anywhere you like
            img_path = os.path.join(os.path.abspath(dst_img_dir), '{}.jpg'.format(h5_id))
            gt_path = os.path.join(os.path.abspath(dst_label_dir), h5_filename)
            train_val_pair_list.append((img_path, gt_path))
        
        mat_list = glob.glob(os.path.join(args.test_img_dir, '*.jpg'))
        for mat_path in mat_list:
            mat_filename = os.path.split(mat_path)[1]
            iid = os.path.splitext(mat_filename)[0]
            img_path = os.path.join(os.path.abspath(args.test_img_dir), '{}.jpg'.format(iid))
            test_list.append(img_path)
            test_iids.append(iid)

        # save to file
        with open(os.path.join(args.output_dir, 'trainval_pair.lst'), 'w') as f:
            for img_path, gt_path in train_val_pair_list:
                f.write('{} {}\n'.format(img_path, gt_path))
        print('Write train list to {}.'.format(os.path.join(args.output_dir, 'trainval_pair.lst')))

        with open(os.path.join(args.output_dir, 'test.lst'), 'w') as f:
            for img_path in test_list:
                f.write('{}\n'.format(img_path))
        print('Write test list to {}.'.format(os.path.join(args.output_dir, 'test.lst')))

        with open(os.path.join(args.output_dir, 'test_ori_iids.lst'), 'w') as f:
            for iid in test_iids:
                f.write('{}\n'.format(iid))
        print('Write test ids to {}.'.format(os.path.join(args.output_dir, 'test_ori_iids.lst')))

    print('Down!')


if __name__ == '__main__':
    main()
