import torch
import torch.nn as nn
from module.baseline.resunet_a.padding import get_padding
import torch.nn.functional as F
import ever as er
from module.loss import SegmentationLoss
from torchvision import models
import segmentation_models_pytorch as smp

def _PSP1x1Conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        # nn.BatchNorm2d(out_channels),
        # norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )

class PSPPooling(nn.Module):
    def __init__(self, in_channels):
        super(PSPPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels)

        self.conv_final = nn.Conv2d(in_channels * 2,in_channels,1)

    def forward(self, x):
        size = x.size()[2:]
        # print(f'input size {size}')
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        
        # print(f'feat1 {feat1.size()}')
        # print(f'feat1 {feat2.size()}')
        # print(f'feat1 {feat3.size()}') 
        # print(f'feat1 {feat4.size()}')

        cat = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        output = self.conv_final(cat)
        # print(f' output size of psp pooling is {output.size()}')
        return output

class ResBlockA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation):
        super(ResBlockA, self).__init__()

        same = get_padding(kernel_size,stride,dilation)
        # same = 1

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=same),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=same))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = same

    def forward(self, x):
        x1 = self.conv_block(x)
        return x1


class ResBlockAD4(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, num):
        super(ResBlockAD4, self).__init__()

        # print(f'in_ch {in_channels} out_ch: {out_channels} dilations: {dilations[0]} Block n th: {num}')

        self.ResBlock1 = ResBlockA(in_channels,
                                   out_channels,
                                   3,
                                   1,
                                   dilation=dilations[0])
        self.ResBlock2 = ResBlockA(in_channels,
                                   out_channels,
                                   3,
                                   1,
                                   dilation=dilations[1])
        self.ResBlock3 = ResBlockA(in_channels,
                                   out_channels,
                                   3,
                                   1,
                                   dilation=dilations[2])
        self.ResBlock4 = ResBlockA(in_channels,
                                   out_channels,
                                   3,
                                   1,
                                   dilation=dilations[3])
        self.num = num

    def forward(self, x):
        x1 = self.ResBlock1(x)
        x2 = self.ResBlock2(x)
        x3 = self.ResBlock3(x)
        x4 = self.ResBlock4(x)

        x12 = x1.add(x2)
        x34 = x3.add(x4)
        xAll = x12.add(x34)

        return xAll


class ResBlockAD3(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ResBlockAD3, self).__init__()
        print()
        self.ResBlock1 = ResBlockA(in_channels, out_channels, 3, 1,
                                   dilations[0])
        self.ResBlock2 = ResBlockA(in_channels, out_channels, 3, 1,
                                   dilations[1])
        self.ResBlock3 = ResBlockA(in_channels, out_channels, 3, 1,
                                   dilations[2])

    def forward(self, x):
        x1 = self.ResBlock1(x)
        x2 = self.ResBlock2(x)
        x3 = self.ResBlock2(x)
        x12 = torch.add(x1, x2)
        xAll = torch.add(x12,x3)
        return xAll


class ResBlockAD1(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ResBlockAD1, self).__init__()
        self.ResBlock1 = ResBlockA(in_channels, out_channels, 3, 1,
                                   dilations[0])

    def forward(self, x):
        x1 = self.ResBlock1(x)
        return x1


class Combine(nn.Module):
    def __init__(self,input_size):
        super(Combine, self).__init__()
        self.input_size = input_size
        # print(f'input size: {input_size}')
        self.convn = nn.Sequential(
            nn.Conv2d(input_size * 2,input_size, 1),
            nn.BatchNorm2d(input_size)
        )

        self.relu = nn.ReLU(True)

    def forward(self, input1, input2):
        input1_relu = self.relu(input1.cuda())
        # print(f'input1 relu size : {input1_relu.size()} input2 size : {input2.size()}')
        input_concat = torch.cat([input1_relu, input2.cuda()],dim=1)
        # print(f'input_concat size : {input_concat.size()}')
        output_conv = self.convn(input_concat.cuda())
        # return output_conv
        return output_conv

@er.registry.MODEL.register()
class ResUNetA(er.ERModule):
    def __init__(self, config):
        super(ResUNetA, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)

        self.conv1 = nn.Conv2d(config.in_channel, 32, 1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 1, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 1, stride=2)
        self.conv5 = nn.Conv2d(256, 512, 1, stride=2)
        self.conv6 = nn.Conv2d(512, 1024, 1, stride=2)
         
        
        self.convup1 = nn.Conv2d(1024, 512, 1, stride=1)
        self.convup2 = nn.Conv2d(512, 256, 1, stride=1)
        self.convup3 = nn.Conv2d(256, 128, 1, stride=1)
        self.convup4 = nn.Conv2d(128, 64, 1, stride=1)
        self.convup5 = nn.Conv2d(64, 32, 1, stride=1)

        self.conv_final = nn.Conv2d(32,config.classes,1,stride=1)


        self.rest_block_1 = ResBlockAD4(32, 32, [1, 3, 15, 31],'down1')
        self.rest_block_2 = ResBlockAD4(64, 64, [1, 3, 15, 31],'down2')
        self.rest_block_3 = ResBlockAD3(128, 128, [1, 3, 15])
        self.rest_block_4 = ResBlockAD3(256, 256, [1, 3, 15])
        self.rest_block_5 = ResBlockAD1(512, 512, [1])
        self.rest_block_6 = ResBlockAD1(1024, 1024, [1])

        self.rest_block_up_1 = ResBlockAD4(32, 32, [1, 3, 15, 31],'up1')
        self.rest_block_up_2 = ResBlockAD4(64, 64, [1, 3, 15, 31],'up2')
        self.rest_block_up_3 = ResBlockAD3(128, 128, [1, 3, 15])
        self.rest_block_up_4 = ResBlockAD3(256, 256, [1, 3, 15])
        self.rest_block_up_5 = ResBlockAD1(512, 512, [1])

        self.PSPPooling = PSPPooling(1024)
        self.PSPPoolingResult = PSPPooling(32)


        self.combine1 = Combine(512)
        self.combine2 = Combine(256)
        self.combine3 = Combine(128)
        self.combine4 = Combine(64)
        self.combine5 = Combine(32)



    def forward(self, x, y=None):


        x_conv_1 = self.conv1(x)
        x_resblock_1 = self.rest_block_1(x_conv_1)
        
        x_conv_2 = self.conv2(x_resblock_1)
        x_resblock_2 = self.rest_block_2(x_conv_2)

        x_conv_3 = self.conv3(x_resblock_2)
        x_resblock_3 = self.rest_block_3(x_conv_3)

        x_conv_4 = self.conv4(x_resblock_3)
        x_resblock_4 = self.rest_block_4(x_conv_4)

        x_conv_5 = self.conv5(x_resblock_4)
        x_resblock_5 = self.rest_block_5(x_conv_5)

        x_conv_6 = self.conv6(x_resblock_5)
        x_resblock_6 = self.rest_block_6(x_conv_6)

        x_pooling_1 = self.PSPPooling(x_resblock_6)
        
        #Decoder part  upsampling with combine
       
       #up5

        x_convup_1 = self.convup1(x_pooling_1)
        x_upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_1)

        x_combine_1 = self.combine1(x_upsampling1, x_resblock_5)
        x_resblockup_5 = self.rest_block_up_5(x_combine_1)

      #up4
        x_convup_2 = self.convup2(x_resblockup_5)
        x_upsampling2 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_2)
        
        x_combine_2 = self.combine2(x_upsampling2, x_resblock_4)
        x_resblockup_4 = self.rest_block_up_4(x_combine_2)



      #up3

        x_convup_3 = self.convup3(x_resblockup_4)
        x_upsampling3 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_3)
        x_combine_3 = self.combine3(x_upsampling3, x_resblock_3)
        x_resblockup_3 = self.rest_block_up_3(x_combine_3)

     #up2
        x_convup_4 = self.convup4(x_resblockup_3)
        x_upsampling4 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_4)
        x_combine_4 = self.combine4(x_upsampling4, x_resblock_2)
        x_resblockup_2 = self.rest_block_up_2(x_combine_4)


      #up1
        x_convup_5 = self.convup5(x_resblockup_2)
        x_upsampling5 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_5)
        x_combine_5 = self.combine5(x_upsampling5, x_resblock_1)
        x_resblockup_1 = self.rest_block_up_1(x_combine_5)

        x_combine_6 = self.combine5(x_resblockup_1, x_conv_1)
        
        x_pooling_2 = self.PSPPoolingResult(x_combine_6)

        x_conv_result = self.conv_final(x_pooling_2)

        if self.training:
            return self.loss(x_conv_result, y['cls'])
        
        return x_conv_result.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            classes=7,
            in_channel=3,
            loss=dict(
                ce=dict()
            )
        ))

# if __name__=="__main__":
#     net=ResUNetA(3,2).cuda()
#     x=torch.randn((1,3,512,512)).cuda()
#     out=net(x)
#     print(out.shape)