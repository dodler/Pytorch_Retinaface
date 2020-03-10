import unittest

import matplotlib.pyplot as plt
import torch
from PIL import Image

from face_cropper import Cropper
import time


class TestFaceCropper(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dst_dim = 720
        self.cropper = Cropper(im_width=self.dst_dim,
                               im_height=self.dst_dim)

    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def test_predict_find_face_path_should_get_expected_data(self):
        # fixme, relative paths
        path = 'resources/IM2019070432CO.jpg'
        result = self.cropper.find_face_path(path=path)[0]
        # expected_result = np.l

        print(result.shape)
        # self.assertTrue(np.allclose(expected_result, result))

        img_plot = Cropper.plot_det(result, path=path)
        plt.imshow(img_plot)
        plt.show()

    def test_find_face_batch(self):
        paths = [
            'resources/IM2019070432CO.jpg', 'resources/joshua_bell.jpg'
        ]

        meta = []
        for path in paths:
            img = Image.open(path)
            meta.append((img.width, img.height))

        meta = torch.tensor(meta)

        imgs = [Cropper.read_tensor(k, size=(self.dst_dim, self.dst_dim)) for k in paths]
        img = torch.cat(imgs, 0)

        result = self.cropper.find_face_batch(img, meta)

        for i, r in enumerate(result):
            img_plot = Cropper.plot_det(r, paths[i])
            plt.imshow(img_plot)
            plt.show()

        # fixme, asserts

    def test_inference_speed(self):
        paths = [
            '/home/lyan/Documents/test_deep_fake.png',
        ]

        meta = []
        for path in paths:
            img = Image.open(path)
            meta.append((img.width, img.height))

        meta = torch.tensor(meta)
        print(meta)
        # imgs = [Cropper.read_tensor(k, size=(self.dst_dim, self.dst_dim)) for k in paths]
        imgs = [Cropper.read_tensor(k, size=(256, 256)) for k in paths]
        img = torch.cat(imgs, 0)
        start = time.time()
        result = self.cropper.find_face_batch(img, meta)

        n = 1
        for i in range(n):
            result = self.cropper.find_face_batch(img, meta)
        end = time.time()

        print(result)
        for i, r in enumerate(result):
            img_plot = Cropper.plot_det(r, paths[0])
            plt.imshow(img_plot)
            plt.show()
        print('spent for ', n, ' iterations ', end - start, ' s')
        print('avg time', (end - start) / n)
