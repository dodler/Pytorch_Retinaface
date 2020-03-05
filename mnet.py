import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, phase='train'):
        super(FPN, self).__init__()
        self.phase = phase
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x1, x2, x3):
        output1 = self.output1(x1)
        output2 = self.output2(x2)
        output3 = self.output3(x3)

        if self.phase == 'train':
            up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
            output2 = output2 + up3
        else:
            output2 = output2 + F.upsample_nearest(output3, size=[16, 16])  # hasrcoded for 256 input image

        output2 = self.merge2(output2)

        if self.phase == 'train':
            up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
            output1 = output1 + up2
        else:
            up2 = F.interpolate(output2, size=[32, 32], mode="nearest")
            output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x1, x2, x3


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, phase='train'):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.phase = phase
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        if self.phase == 'train':
            # out = out.permute(0, 2, 3, 1).contiguous()
            return out.reshape(out.shape[0], -1, 2)
        else:
            return out


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, phase='train'):
        super(BboxHead, self).__init__()
        self.phase = phase
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        if self.phase == 'train':
            # out = out.permute(0, 2, 3, 1).contiguous()
            return out.view(out.shape[0], -1, 4)
        else:
            return out
        #


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, phase='train'):
        super(LandmarkHead, self).__init__()
        self.phase = phase
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        if self.phase == 'train':
            # out = out.permute(0, 2, 3, 1).contiguous()
            return out.view(out.shape[0], -1, 10)
        else:
            return out


class RetinaFaceMnet(nn.Module):
    def __init__(self, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFaceMnet, self).__init__()
        self.phase = phase
        self.backbone = MobileNetV1()

        in_channels_list = [64, 128, 256, ]
        out_channels = 64
        self.fpn = FPN(in_channels_list, out_channels, phase=phase)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channels, phase=phase)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channels, phase=phase)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channels, phase=phase)

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2, phase='train'):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num, phase=phase))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2, phase='train'):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num, phase=phase))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2, phase='train'):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num, phase=phase))
        return landmarkhead

    def forward(self, inputs):
        x1, x2, x3 = self.backbone(inputs)
        f1, f2, f3 = self.fpn(x1, x2, x3)
        feature1 = self.ssh1(f1)
        feature2 = self.ssh2(f2)
        feature3 = self.ssh3(f3)

        if self.phase == 'train':
            features = [feature1, feature2, feature3]

            bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
            classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
            ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

            output = (bbox_regressions, classifications, ldm_regressions)
            return output
        else:
            return self.BboxHead[0](feature1), \
                   F.softmax(self.ClassHead[0](feature1), dim=-1), \
                   self.LandmarkHead[0](feature1), \
                   self.BboxHead[1](feature2), \
                   F.softmax(self.ClassHead[1](feature2), dim=-1), \
                   self.LandmarkHead[1](feature2), \
                   self.BboxHead[2](feature3), \
                   F.softmax(self.ClassHead[2](feature3), dim=-1), \
                   self.LandmarkHead[2](feature3)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    net = RetinaFaceMnet()
    print(net(torch.randn(1, 3, 256, 256)))
