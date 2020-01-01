# This file is not used by the tracking application
import torch
import torch.nn as nn

class DSConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernels_per_layer=4, groups=1):
        super(DSConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes),
            nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_channels, residual_in_channels, out_channels, size):
        super(UNetUp, self).__init__()
        self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        self.conv = DSConv2d(in_channels + residual_in_channels, out_channels, 1, 1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# Copy of torchvision ShuffleNetV2 for type annotation
def channel_shuffle(x, groups: int):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        self.branch1 = nn.Sequential(
            self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

# Facial landmark detection model
class OpenSeeNetSNV2(ShuffleNetV2):
    def __init__(self):
        super(OpenSeeNetSNV2, self).__init__([4, 8, 4], [24, 116, 232, 464, 1024])
        self.up1 = UNetUp(1024, 232, 256, (14,14))
        self.up2 = UNetUp(256, 116, 198 * 1, (28,28))
        self.group = DSConv2d(198 * 1, 198, kernels_per_layer=4, groups=3)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        r2 = x
        x = self.stage3(x)
        r3 = x
        x = self.stage4(x)
        x = self.conv5(x)

        x = self.up1(x, r3)
        x = self.up2(x, r2)
        x = self.group(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

# Adaptive Wing Loss with offset layers and emphasis for eyes and eyebrows with 66 landmark points
def AdapWingLoss(pre_hm, gt_hm):
    # pre_hm = pre_hm.to('cpu')
    # gt_hm = gt_hm.to('cpu')
    theta = 0.5
    alpha = 2.1
    w = 14
    e = 1
    A = w * (1 / (1 + torch.pow(theta / e, alpha - gt_hm))) * (alpha - gt_hm) * torch.pow(theta / e, alpha - gt_hm - 1) * (1 / e)
    C = (theta * A - w * torch.log(1 + torch.pow(theta / e, alpha - gt_hm)))

    batch_size = gt_hm.size()[0]
    hm_num = gt_hm.size()[1]

    mask = torch.zeros_like(gt_hm)
    # W = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    first = True
    first_mask = None
    for i in range(batch_size):
        img_list = []
        for j in range(hm_num // 3):
            hm = np.round(gt_hm[i][j].cpu().numpy() * 255)
            img_list.append(hm)
        img_merge = cv2.merge(img_list)
        img_dilate = cv2.morphologyEx(img_merge, cv2.MORPH_DILATE, kernel)
        img_dilate[img_dilate < 51] = 1  # 0*W+1
        img_dilate[img_dilate >= 51] = 11  # 1*W+1
        img_dilate = np.array(img_dilate, dtype=np.int)
        img_dilate = img_dilate.transpose(2, 0, 1)
        dilated = torch.from_numpy(img_dilate).float()
        if first:
            first_mask = dilated
        first = False
        dilated[17:27] *= 1.2
        dilated[17] *= 1.3
        dilated[18] *= 1.4
        dilated[25] *= 1.4
        dilated[26] *= 1.3
        dilated[36:47] *= 2.5
        mask[i] = torch.cat([dilated, dilated, dilated], 0)

    diff_hm = torch.abs(gt_hm - pre_hm)
    AWingLoss = A * diff_hm - C
    idx = diff_hm < theta
    AWingLoss[idx] = w * torch.log(1 + torch.pow(diff_hm / e, alpha - gt_hm))[idx]

    AWingLoss *= mask
    sum_loss = torch.sum(AWingLoss)
    all_pixel = torch.sum(mask)
    mean_loss = sum_loss / all_pixel

    return first_mask.detach(), mean_loss

# The fast model runs on grayscale 112x112 using the ShuffleNet V2 0.5x configuration with input_channels=1. It was trained using AWL, but without the additional weighting for certain features
.