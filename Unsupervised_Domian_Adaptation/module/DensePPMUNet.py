import torch.nn as nn
import torch
from module.DensePPM import DensePPM
from module.PPM import PPM



class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层, out_2输入到下一层.
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class DensePPMUNet(nn.Module):
    def __init__(self,in_channel=3, n_classes = 3,ppm="DensePPM",pool_size=[2,3,4,6]):
        super(DensePPMUNet, self).__init__()
        #out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        self.n_classes=n_classes
        # out_channels=[64, 128, 256, 512, 1024]
        out_channels=[32, 64, 128, 256, 512]
        #下采样
        self.d1=DownsampleLayer(in_channel,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #金字塔池化
        if ppm == "DensePPM":
            self.PPMLayer = DensePPM(in_channels=out_channels[1], num_classes=n_classes, reduction_dim=64, pool_sizes=pool_size)
        elif ppm == "PPM":
            self.PPMLayer = PPM(in_dim=out_channels[1], reduction_dim=32, bins=pool_size)

        
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.PPMLayer(out8)
        if self.training:
            return out
        else:
            return out.softmax(dim=1)
    #权值初始化
    # def initialize_weights(self, *models):
    #     for model in models:
    #         for m in model.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.weight.data.fill_(1.)
    #                 m.bias.data.fill_(1e-4)
    #             elif isinstance(m, nn.Linear):
    #                 m.weight.data.normal_(0.0, 0.0001)
    #                 m.bias.data.zero_()

# if __name__ == '__main__':
#     test_data=torch.randn(2,3,256,256)
#     net=DensePPMUNet()
#     out=net(test_data)
#     print(out.shape)








