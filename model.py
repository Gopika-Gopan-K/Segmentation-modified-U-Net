import torch
import torch.nn as nn
from torch.nn import Module

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Two sequential convolution section of Unet
def doubleconv(inp, out):
    double_conv = nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3),
        nn.BatchNorm2d(out,track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out, out, kernel_size=3),
        nn.BatchNorm2d(out,track_running_stats=False),
        nn.ReLU(inplace=True)
    )
    return double_conv


# Crop the encoder feature to the size of corresponding decoder for concatenation
def crop_feat(input_tensor, target_tensor):
    out_size = target_tensor.size()[2]
    inp_size = input_tensor.size()[2]
    delta = (inp_size-out_size) // 2
    if (inp_size-out_size) % 2 == 0:
        result = input_tensor[:, :, delta:inp_size-delta, delta:inp_size-delta]
    else:
        result = input_tensor[:, :, delta:inp_size-delta-1, delta:inp_size-delta-1]

    return result


# Spatial Channel Attention Block
class sca(Module):
    def __init__(self, inp):
        super(sca, self).__init__()
        self.c_attn_conv = nn.Sequential(nn.Conv2d(inp, inp // 16, 1, bias=False),
                                         nn.ReLU(),
                                         nn.Conv2d(inp // 16, inp, 1, bias=False)
                                         )
        self.c_sig = nn.Sigmoid()

        self.avg_ch = nn.AdaptiveAvgPool2d(1)
        self.max_ch = nn.AdaptiveMaxPool2d(1)
        self.s_attn_conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=7 // 2, bias=False),
                                         nn.Sigmoid())

    def forward(self, input_tensor):
        # Channel Attention
        avg_ch_pool = self.avg_ch(input_tensor)
        max_ch_pool = self.max_ch(input_tensor)

        out_1 = self.c_attn_conv(avg_ch_pool)
        out_2 = self.c_attn_conv(max_ch_pool)

        c_sum = out_1+out_2
        ch_out = self.c_sig(c_sum)
        input_tensor = input_tensor * ch_out
        # Spatial Attention
        avg_pool = torch.mean(input_tensor, dim=1, keepdim=True)
        max_pool = torch.max(input_tensor, dim=1, keepdim=True)
        x = torch.cat([avg_pool, max_pool.values], dim=1)

        x = self.s_attn_conv(x)
        x = torch.mul(input_tensor, x)

        return x


# Function to apply spatial channel attention block
def spatial_channel_attn(input_tensor):
    input_tensor = input_tensor.type(torch.cuda.FloatTensor)
    inp = input_tensor.size()[1]

    sca_model = sca(inp).to(device)
    x = sca_model(input_tensor)

    return x


# Atrous spatial pyramid pooling block
class aspp(Module):
    def __init__(self, inp, out):
        super(aspp, self).__init__()
        self.aconv0 = nn.Sequential(nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                    nn.BatchNorm2d(out,track_running_stats=False),
                                    nn.ReLU(inplace=True))
        self.aconv1 = nn.Sequential(nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
                                    nn.BatchNorm2d(out,track_running_stats=False),
                                    nn.ReLU(inplace=True))
        self.aconv2 = nn.Sequential(nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
                                    nn.BatchNorm2d(out,track_running_stats=False),
                                    nn.ReLU(inplace=True))
        self.aconv3 = nn.Sequential(nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                                    nn.BatchNorm2d(out,track_running_stats=False),
                                    nn.ReLU(inplace=True))
        self.aconv4 = nn.Sequential(nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                    nn.BatchNorm2d(out),
                                    nn.ReLU(inplace=True))
        self.final_conv = nn.Sequential(
            nn.Conv2d(out * 5, inp, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True))

    def forward(self, input_tensor):
        x0 = self.aconv0(input_tensor)
        x1 = self.aconv1(input_tensor)
        x2 = self.aconv2(input_tensor)
        x3 = self.aconv3(input_tensor)
        x4 = self.aconv4(input_tensor)
        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        aspp_out = self.final_conv(x)

        return aspp_out


# Function to carry out atrous spatial pyramid pooling (In our case input and output is made to be of same size)
def atrous_spatial_pyramid_pooling(input_tensor):
    input_tensor = input_tensor.type(torch.cuda.FloatTensor)
    inp = input_tensor.size()[1]
    out = inp // 4

    asppmodel = aspp(inp, out).to(device)
    aspp_out = asppmodel(input_tensor)
    res_aspp = aspp_out+input_tensor
    return res_aspp


# Unet Model
class Unet(Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.DownConv1 = doubleconv(1, 64)
        self.DownConv2 = doubleconv(64, 128)
        self.DownConv3 = doubleconv(128, 256)
        self.DownConv4 = doubleconv(256, 512)
        self.DownConv5 = doubleconv(512, 1024)

        self.UpTrans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.UpTrans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.UpTrans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.UpTrans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.UpConv1 = doubleconv(1024, 512)
        self.UpConv2 = doubleconv(512, 256)
        self.UpConv3 = doubleconv(256, 128)
        self.UpConv4 = doubleconv(128, 64)

        self.out1 = nn.Conv2d(64, 1, kernel_size=1)
        self.out2 = nn.Conv2d(128, 1, kernel_size=1)
        self.out3 = nn.Conv2d(256, 1, kernel_size=1)
        self.out4 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        # x=(batch size, channel, height, width)
        # encoder

        x1 = self.DownConv1(x)
        x2 = self.maxpool2x2(x1)
        x3 = self.DownConv2(x2)
        x4 = self.maxpool2x2(x3)
        x5 = self.DownConv3(x4)
        x6 = self.maxpool2x2(x5)
        x7 = self.DownConv4(x6)
        x8 = self.maxpool2x2(x7)
        x9 = self.DownConv5(x8)

        # ASPP
        x9_aspp = atrous_spatial_pyramid_pooling(x9)

        # decoder
        z1 = self.UpTrans1(x9_aspp)
        y1 = crop_feat(x7, z1)

        x10 = self.UpConv1(spatial_channel_attn(torch.cat([y1, z1], 1)))

        z2 = self.UpTrans2(x10)
        y2 = crop_feat(x5, z2)
        x11 = self.UpConv2(spatial_channel_attn(torch.cat([y2, z2], 1)))

        z3 = self.UpTrans3(x11)
        y3 = crop_feat(x3, z3)
        x12 = self.UpConv3(spatial_channel_attn(torch.cat([y3, z3], 1)))

        z4 = self.UpTrans4(x12)
        y4 = crop_feat(x1, z4)
        x13 = self.UpConv4(spatial_channel_attn(torch.cat([y4, z4], 1)))

        # ASPP
        x13_aspp = atrous_spatial_pyramid_pooling(x13)
        out1 = self.out1(x13_aspp)
        out2 = self.out2(x12)
        out3 = self.out3(x11)
        out4 = self.out4(x10)
        return out1, out2, out3, out4

