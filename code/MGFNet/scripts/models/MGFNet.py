import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbone import BN_MOMENTUM, hrnet_classification
from .resnet import resnet101
from math import sqrt

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return y

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MGFM(nn.Module):
    def __init__(self, in_channels):
        super(MGFM, self).__init__()
        self.in_channel = in_channels
        self.eca_x = eca_block(channel=self.in_channel)
        self.eca_y = eca_block(channel=self.in_channel)
        self.mlp_x = Mlp(in_features=self.in_channel * 2, out_features=self.in_channel)
        self.mlp_y = Mlp(in_features=self.in_channel * 2, out_features=self.in_channel)
        self.sigmoid = nn.Sigmoid()

        self.mlp = Mlp(in_features= in_channels,out_features=in_channels)

    def forward(self, opt, sar):

        # Fusion-Stage-1 ECA Channel Attention
        w_opt = self.eca_x(opt)
        w_sar = self.eca_y(sar)
        N, C, H, W = w_opt.shape

        w_opt = torch.flatten(w_opt, 1)
        w_sar = torch.flatten(w_sar, 1)

        w = torch.concat([w_opt, w_sar], 1)
        
        # Fusion-Stage-2 MLP
        w1 = self.mlp_x(w)
        w1 = self.sigmoid(w1.reshape([N, self.in_channel, H, W]))

        w2 = self.mlp_y(w)
        w2 = self.sigmoid(w2.reshape([N, self.in_channel, H, W]))

        # Gating-Stage
        out1 = opt * w1
        out2 = sar * w2
        f = torch.cat((out1,out2),1)

        return f

class HRnet_Backbone(nn.Module):
    def __init__(self, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet_Backbone, self).__init__()
        self.model    = hrnet_classification(backbone = backbone, pretrained = pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        
        return y_list

class MGFNet(nn.Module):
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w48', pretrained = False):
        super(MGFNet, self).__init__()
        self.opt_encoder       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)
        self.sar_encoder       = resnet101(pretrained)
        last_inp_channels   = np.sum(self.opt_encoder.model.pre_stage_channels)

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

        layer = [24, 48, 96, 192]
        self.fusion_1 = MGFM(layer[0])
        self.fusion_2 = MGFM(layer[1])
        self.fusion_3 = MGFM(layer[2])
        self.fusion_4 = MGFM(layer[3])

        self.shortcut_conv1 = nn.Sequential(
            nn.Conv2d(384, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv2 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv3 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv4 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv1_1 = nn.Sequential(
            nn.Conv2d(2048, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv2_1 = nn.Sequential(
            nn.Conv2d(1024, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv3_1 = nn.Sequential(
            nn.Conv2d(512, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv4_1 = nn.Sequential(
            nn.Conv2d(256, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
    def forward(self, opt,sar):

        H, W = opt.size(2), opt.size(3)

        # MFE Network
        [feat1_opt, feat2_opt, feat3_opt, feat4_opt] = self.opt_encoder(opt)
        [feat1_sar, feat2_sar, feat3_sar, feat4_sar, feat5_sar] = self.sar_encoder(sar)

        # Shortcut Layer
        feat4_opt = self.shortcut_conv1(feat4_opt)
        feat3_opt = self.shortcut_conv2(feat3_opt)
        feat2_opt = self.shortcut_conv3(feat2_opt)
        feat1_opt = self.shortcut_conv4(feat1_opt)

        feat5_sar = self.shortcut_conv1_1(feat5_sar)
        feat4_sar = self.shortcut_conv2_1(feat4_sar)
        feat3_sar = self.shortcut_conv3_1(feat3_sar)
        feat2_sar = self.shortcut_conv4_1(feat2_sar)

        # MGFM Fusion
        fusion_feat1 = self.fusion_1(feat1_opt, feat2_sar)
        fusion_feat2 = self.fusion_2(feat2_opt, feat3_sar)
        fusion_feat3 = self.fusion_3(feat3_opt, feat4_sar)
        fusion_feat4 = self.fusion_4(feat4_opt, feat5_sar)

        # Decoder
        x0_h, x0_w = fusion_feat1.size(2), fusion_feat1.size(3)
        x1 = F.interpolate(fusion_feat2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(fusion_feat3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(fusion_feat4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([fusion_feat1, x1, x2, x3], 1)
        x = self.last_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x

    def freeze_backbone(self):
        for param in self.opt_encoder.parameters():
            param.requires_grad = False
        # for param in self.sar_encoder.parameters():
        #     param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.opt_encoder.parameters():
            param.requires_grad = True
        # for param in self.sar_encoder.parameters():
        #     param.requires_grad = True

if __name__ == "__main__":
    model = MGFNet(num_classes=8)
    model.train()
    sar = torch.randn(2, 3, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", model(opt, sar).shape)