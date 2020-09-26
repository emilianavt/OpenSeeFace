# This file is not used by the tracking application and currently outdated
import torch
import torch.nn as nn
import geffnet.mobilenetv3 # geffnet.mobilenetv3._gen_mobilenet_v3 needs to be patched to return the parameters instead of instantiating the network
from geffnet.efficientnet_builder import round_channels

class DSConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernels_per_layer=4, groups=1, old=0):
        super(DSConv2d, self).__init__()
        if old == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups)
            )
        elif old == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes * kernels_per_layer),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes * kernels_per_layer),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
            )
    def forward(self, x):
        x = self.conv(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_channels, residual_in_channels, out_channels, size, old=0):
        super(UNetUp, self).__init__()
        self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        self.conv = DSConv2d(in_channels + residual_in_channels, out_channels, 1, 1, old=old)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# This is the gaze tracking model
class OpenSeeFaceGaze(geffnet.mobilenetv3.MobileNetV3):
    def __init__(self):
        kwargs = geffnet.mobilenetv3._gen_mobilenet_v3(['small'])
        super(OpenSeeFaceGaze, self).__init__(**kwargs)
        self.up1 = UNetUp(576, 48, 64, (2,2), old=2)
        self.up2 = UNetUp(64, 24, 32, (4,4), old=2)
        self.up3 = UNetUp(32, 16, 15, (8,8), old=2)
        self.group = DSConv2d(15, 3, kernels_per_layer=4, groups=3, old=2)
    def _forward_impl(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        r1 = None
        r2 = None
        r3 = None
        for i, feature in enumerate(self.blocks):
            x = feature(x)
            if i == 3:
                r3 = x
            if i == 1:
                r2 = x
            if i == 0:
                r1 = x
        x = self.up1(x, r3)
        x = self.up2(x, r2)
        x = self.up3(x, r1)
        x = self.group(x)
        return x
    def forward(self, x):
        return self._forward_impl(x)

# This is the face detection model. Because the landmark model is very robust, it gets away with predicting very rough bounding boxes. It is fully convolutional and can be made to run on different resolutions. It was trained on 224x224 crops and the most reasonable results can be found in the range of 224x224 to 640x640.
class OpenSeeFaceDetect(geffnet.mobilenetv3.MobileNetV3):
    def __init__(self, size="large", channel_multiplier=0.1):
        kwargs = geffnet.mobilenetv3._gen_mobilenet_v3([size], channel_multiplier=channel_multiplier)
        super(OpenSeeFaceDetect, self).__init__(**kwargs)
        if size == "large":
            self.up1 = UNetUp(round_channels(960, channel_multiplier), round_channels(112, channel_multiplier), 256, (14,14), old=1)
            self.up2 = UNetUp(256, round_channels(40, channel_multiplier), 128, (28,28), old=1)
            self.up3 = UNetUp(128, round_channels(24, channel_multiplier), 64, (56,56), old=1)
            self.group = DSConv2d(64, 2, kernels_per_layer=4, groups=2, old=1)
            self.r1_i = 1
            self.r2_i = 2
            self.r3_i = 4
        elif size == "small":
            self.up1 = UNetUp(round_channels(576, channel_multiplier), round_channels(40, channel_multiplier), 256, (14,14), old=1)
            self.up2 = UNetUp(256, round_channels(24, channel_multiplier), 128, (28,28), old=1)
            self.up3 = UNetUp(128, round_channels(16, channel_multiplier), 64, (56,56), old=1)
            self.group = DSConv2d(64, 2, kernels_per_layer=4, groups=2, old=1)
            self.r1_i = 0
            self.r2_i = 1
            self.r3_i = 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, dilation=1, stride=1, padding=1)
    def _forward_impl(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        r2 = None
        r3 = None
        for i, feature in enumerate(self.blocks):
            x = feature(x)
            if i == self.r3_i:
                r3 = x
            if i == self.r2_i:
                r2 = x
            if i == self.r1_i:
                r1 = x
        x = self.up1(x, r3)
        x = self.up2(x, r2)
        x = self.up3(x, r1)
        x = self.group(x)
        x2 = self.maxpool(x)
        return x, x2
    def forward(self, x):
        return self._forward_impl(x)

def logit_arr(p, factor=16.0):
    p = p.clamp(0.0000001, 0.9999999)
    return torch.log(p / (1 - p)) / factor

# Landmark detection model
# Models:
# 0: "small", 0.5
# 1: "small", 1.0
# 2: "large", 0.75
# 3: "large", 1.0
class OpenSeeFaceLandmarks(geffnet.mobilenetv3.MobileNetV3):
    def __init__(self, size="large", channel_multiplier=1.0, inference=False):
        kwargs = geffnet.mobilenetv3._gen_mobilenet_v3([size], channel_multiplier=channel_multiplier)
        super(OpenSeeFaceLandmarks, self).__init__(**kwargs)
        if size == "large":
            self.up1 = UNetUp(round_channels(960, channel_multiplier), round_channels(112, channel_multiplier), 256, (14,14))
            self.up2 = UNetUp(256, round_channels(40, channel_multiplier), 198 * 1, (28,28))
            self.group = DSConv2d(198 * 1, 198, kernels_per_layer=4, groups=3)
            self.r2_i = 2
            self.r3_i = 4
        elif size == "small":
            self.up1 = UNetUp(round_channels(576, channel_multiplier), round_channels(40, channel_multiplier), 256, (14,14))
            self.up2 = UNetUp(256, round_channels(24, channel_multiplier), 198 * 1, (28,28))
            self.group = DSConv2d(198 * 1, 198, kernels_per_layer=4, groups=3)
            self.r2_i = 1
            self.r3_i = 2
        self.inference = inference
    def _forward_impl(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        r2 = None
        r3 = None
        for i, feature in enumerate(self.blocks):
            x = feature(x)
            if i == self.r3_i:
                r3 = x
            if i == self.r2_i:
                r2 = x
        x = self.up1(x, r3)
        x = self.up2(x, r2)
        x = self.group(x)

        if self.inference:
            t_main = x[:, 0:66].reshape((-1, 66, 28*28))
            t_m = t_main.argmax(dim=2)
            indices = t_m.unsqueeze(2)
            t_conf = t_main.gather(2, indices).squeeze(2)
            t_off_x = x[:, 66:132].reshape((-1, 66, 28*28)).gather(2, indices).squeeze(2)
            t_off_y = x[:, 132:198].reshape((-1, 66, 28*28)).gather(2, indices).squeeze(2)
            t_off_x = (223. * logit_arr(t_off_x) + 0.5).floor()
            t_off_y = (223. * logit_arr(t_off_y) + 0.5).floor()
            t_x = 223. * (t_m / 28.).floor() / 27. + t_off_x
            t_y = 223. * t_m.remainder(28.).float() / 27. + t_off_y
            x = (t_conf.mean(1), torch.stack([t_x, t_y, t_conf], 2))

        return x
    def forward(self, x):
        return self._forward_impl(x)

# lm_modelT for 56x56 30 point inference
class OpenSeeFaceLandmarks30Pt(geffnet.mobilenetv3.MobileNetV3):
    def __init__(self, size="large", channel_multiplier=1.0, inference=False):
        kwargs = geffnet.mobilenetv3._gen_mobilenet_v3([size], channel_multiplier=channel_multiplier)
        super(OpenSeeFaceLandmarks30Pt, self).__init__(**kwargs)
        self.up1 = UNetUp(960, 112, 256, (4,4))
        self.up2 = UNetUp(256, 40, 180, (7,7))
        self.group = DSConv2d(180, 90, kernels_per_layer=4, groups=3)
        self.inference = inference
    def _forward_impl(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        r2 = None
        r3 = None
        for i, feature in enumerate(self.blocks):
            x = feature(x)
            if i == 4:
                r3 = x
            if i == 2:
                r2 = x
        x = self.up1(x, r3)
        x = self.up2(x, r2)
        x = self.group(x)

        if self.inference:
            t_main = x[:, 0:30].reshape((-1, 30, 7*7))
            t_m = t_main.argmax(dim=2)
            indices = t_m.unsqueeze(2)
            t_conf = t_main.gather(2, indices).squeeze(2)
            t_off_x = x[:, 30:60].reshape((-1, 30, 7*7)).gather(2, indices).squeeze(2)
            t_off_y = x[:, 60:90].reshape((-1, 30, 7*7)).gather(2, indices).squeeze(2)
            t_off_x = 55. * logit_arr(t_off_x, factor=8.0)
            t_off_y = 55. * logit_arr(t_off_y, factor=8.0)
            t_x = 55. * (t_m / 7.).floor() / 6. + t_off_x
            t_y = 55. * t_m.remainder(7.).float() / 6. + t_off_y
            x = (t_conf.mean(1), torch.stack([t_x, t_y, t_conf], 2))

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
        # These weights varied between training runs and model sizes
        dilated[17:27][dilated[17:27] > 1] *= 1.4
        dilated[17, dilated[17] > 1] *= 1.6
        dilated[18, dilated[18] > 1] *= 1.8
        dilated[25, dilated[25] > 1] *= 1.8
        dilated[26, dilated[26] > 1] *= 1.6
        dilated[36:48][dilated[36:48] > 1] *= 2.8
        # Used for a very small model
        #dilated[[37,38,40,41,43,44,46,47]][dilated[[37,38,40,41,43,44,46,47]] > 1] *= 20.8
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



# Checkpoint test
if __name__== "__main__":
    print("Checking gaze model")
    m=OpenSeeFaceGaze()
    ckpt = torch.load("gaze.pth")
    m.load_state_dict(ckpt)
    print("Checking detection model")
    m=OpenSeeFaceDetect()
    ckpt = torch.load("detection.pth")
    m.load_state_dict(ckpt)
    print("Checking lm_model0 model")
    m=OpenSeeFaceLandmarks("small", 0.5)
    ckpt = torch.load("lm_model0.pth")
    m.load_state_dict(ckpt)
    print("Checking lm_model1 model")
    m=OpenSeeFaceLandmarks("small", 1.0)
    ckpt = torch.load("lm_model1.pth")
    m.load_state_dict(ckpt)
    print("Checking lm_model2 model")
    m=OpenSeeFaceLandmarks("large", 0.75)
    ckpt = torch.load("lm_model2.pth")
    m.load_state_dict(ckpt)
    print("Checking lm_model3 model")
    m=OpenSeeFaceLandmarks("large", 1.0)
    ckpt = torch.load("lm_model3.pth")
    m.load_state_dict(ckpt)
    print("Checking lm_modelT model")
    m=OpenSeeFaceLandmarks("large", 1.0)
    ckpt = torch.load("lm_modelT.pth")
    m.load_state_dict(ckpt)
