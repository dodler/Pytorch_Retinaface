import unittest

import torch

from models.net import FPN


class TestFpn(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)
        self.fpn = FPN(10, 20)

    def test_simple_run(self):
        inp = torch.randn(2, 3, 20, 20)
        out = self.fpn(inp)
        print(out.shape)
