import pickle
import os.path as osp

import cv2
import numpy as np
import torch


class PckDataset:
    def __init__(self, pck_path, dst_dim):
        self.dst_dim = dst_dim
        with open(pck_path, 'rb') as f:
            data = pickle.load(f)

        self.pck_imgs = []
        for img_dir in data:
            imgs = data[img_dir]
            imgs = [osp.join(img_dir, k) for k in imgs]
            self.pck_imgs.extend(imgs)

    def __len__(self):
        return len(self.pck_imgs)

    def __getitem__(self, item):
        img_path = self.pck_imgs[item]
        img_raw = cv2.imread(img_path)
        im_height, im_width, _ = img_raw.shape
        img_raw = cv2.resize(img_raw, (self.dst_dim, self.dst_dim))
        img = np.float32(img_raw)

        # im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor([im_width, im_height])