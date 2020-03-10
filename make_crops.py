import argparse
import os.path as osp
import pickle

from tqdm import *

from face_cropper import Cropper

import torch

from pck_dataset import PckDataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Retina face cropper')

parser.add_argument('--pck-path', type=str, required=True)
parser.add_argument('--network', type=str, required=False, default='mobile0.25')
parser.add_argument('--weights', type=str, required=False, default='weights/mobilenet0.25_Final.pth')
parser.add_argument('--batchsize', type=int, required=False, default=128)
parser.add_argument('--num-workers', type=int, required=False, default=8)
parser.add_argument('--dst-dim', type=int, required=False, default=512)

args = parser.parse_args()
cropper = Cropper(network=args.network, weights_path=args.weights,
                  im_height=args.dst_dim, im_width=args.dst_dim)

dataset = PckDataset(args.pck_path, dst_dim=args.dst_dim)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batchsize,
                                         num_workers=args.num_workers,
                                         shuffle=False)

result = []
for batch in tqdm(dataloader):
    img_tensor, meta = batch

    img_tensor = img_tensor.to(0)

    det = cropper.find_face_batch(img_tensor=img_tensor, orig_size=meta)
    result.append(det)

with open('{}.crops.pck'.format(args.pck_path), 'wb') as f:
    pickle.dump(result, f)
