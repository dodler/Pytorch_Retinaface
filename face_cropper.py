from __future__ import print_function

import time

import cv2
import numpy as np
import torch
from numba import jit

from data import cfg_mnet, cfg_re50
from detect import load_model
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

torch.set_grad_enabled(False)

resize = 1
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750


# @jit("(float32[:],int32,int32,float32[:])")
@jit(nopython=True)
def scale_result(result, im_height, im_width, meta):
    h = im_height / meta[1]
    w = im_width / meta[0]

    for j in np.arange(result.shape[0]):
        result[j][0] /= w
        result[j][2] /= w
        result[j][5] /= w
        result[j][7] /= w
        result[j][9] /= w
        result[j][11] /= w
        result[j][13] /= w

        result[j][1] /= h
        result[j][3] /= h
        result[j][6] /= h
        result[j][8] /= h
        result[j][10] /= h
        result[j][12] /= h
        result[j][14] /= h
    return result


class Cropper:
    def __init__(self,
                 im_width,
                 im_height,
                 network='mobile0.25',
                 weights_path='/Users/a16976500/Documents/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth',
                 ):
        if network == "mobile0.25":
            cfg = cfg_mnet
        elif network == "resnet50":
            cfg = cfg_re50
        else:
            raise Exception('Network {} is not supported'.format(network))

        self.cfg = cfg
        self.variance = torch.tensor(self.cfg['variance'])
        # self.variance = self.cfg['variance']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        cpu = not torch.cuda.is_available()

        net = RetinaFace(cfg=cfg, phase='test')
        net = load_model(net, weights_path, cpu)
        net.eval()
        print(net)
        print('Finished loading model!')
        if not cpu:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        self.net = net.to(self.device)

    @staticmethod
    def plot_det(dets, path, vis_thres=0.6):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = img.copy()
        for det in dets:
            if det[4] < vis_thres:
                continue
            text = "{:.4f}".format(det[4])
            det = list(map(int, det))

            cv2.rectangle(result, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)
            cx = det[0]
            cy = det[1] + 12
            cv2.putText(result, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(result, (det[5], det[6]), 1, (0, 0, 255), 4)
            cv2.circle(result, (det[7], det[8]), 1, (0, 255, 255), 4)
            cv2.circle(result, (det[9], det[10]), 1, (255, 0, 255), 4)
            cv2.circle(result, (det[11], det[12]), 1, (0, 255, 0), 4)
            cv2.circle(result, (det[13], det[14]), 1, (255, 0, 0), 4)
        return result

    def find_face_path(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        return self.find_face_npimg(img_raw=img)

    @staticmethod
    def read_tensor(img_path, size=None):
        img_raw = cv2.imread(img_path)
        if size is not None:
            img_raw = cv2.resize(img_raw, size)
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).unsqueeze(0)

    def find_face_batch(self, img_tensor, img_meta):
        _, _, h, w = img_tensor.shape
        scale = torch.Tensor([img_tensor.shape[2],
                              img_tensor.shape[3],
                              img_tensor.shape[2],
                              img_tensor.shape[3]])

        img = img_tensor.to(self.device)
        scale = scale.to(self.device)
        loc, conf, landms = self.net(img)
        return self.__postprocess__(im_height=h, im_width=w, loc=loc, scale=scale,
                                    img=img, conf=conf, landms=landms, meta=img_meta)

    def find_face_npimg(self, img_raw):
        img = img_raw.astype(np.float32)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)

        return self.__postprocess__(im_height=im_height, im_width=im_width,
                                    loc=loc, scale=scale, img=img,
                                    conf=conf, landms=landms)

    def __postprocess__(self, im_height, im_width, loc, scale, img, conf, landms, meta=None):

        n_res = loc.shape[0]
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        result = []
        for i in range(n_res):
            boxes = decode(loc[i, :, :], prior_data, self.variance)
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf[i].data.cpu().numpy()[:, 1]
            landmarks = decode_landm(landms[i], prior_data, self.variance)
            landmarks = landmarks * scale1 / resize
            landmarks = landmarks.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            landmarks = landmarks[inds]
            scores = scores[inds]

            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep, :]
            landmarks = landmarks[keep]

            dets = dets[:keep_top_k, :]
            landmarks = landmarks[:keep_top_k, :]
            fdet = np.concatenate((dets, landmarks), axis=1)
            if meta is not None:
                scale_result(fdet, im_height=im_height, im_width=im_width,
                                    meta=meta.numpy().astype(np.float32)[i, :])
            result.append(fdet)

        return result
