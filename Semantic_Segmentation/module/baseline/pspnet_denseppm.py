import torch
from torch import nn
from torch.nn import functional as F
import ever as er
from module.baseline.base_resnet.resnet import ResNetEncoder
from module.loss import SegmentationLoss
from module.baseline.denseppm import DensePPM


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)

class PSPDownsample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(PSPDownsample, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.Conv_BN_ReLU_2(x)
        return x

@er.registry.MODEL.register()
class PSPNet_DensePPM(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        # self.feats = getattr(extractors, backend)(pretrained)
        self.feats = ResNetEncoder(self.config.encoder)

        self.down_1 = PSPDownsample(256, 128)
        self.down_2 = PSPDownsample(512, 256)
        self.down_3 = PSPDownsample(1024, 512)
        self.down_4 = PSPDownsample(2048, 1024)
        self.down_5 = PSPDownsample(128+256+512+1024, self.config.psp.psp_size)

        self.psp = DensePPM(in_channels = self.config.psp.psp_size, reduction_dim = 64, pool_sizes = self.config.psp.sizes)
        psp_out_dim = self.config.psp.psp_size + 64*((len(self.config.psp.sizes)*len(self.config.psp.sizes) + len(self.config.psp.sizes))//2)
        self.drop_1 = nn.Dropout2d(p=0.3)


        self.up_1 = PSPUpsample(psp_out_dim, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Conv2d(64, self.config.classes, kernel_size=1)
        self.loss = SegmentationLoss(self.config.loss)

    def forward(self, x, y=None):
        h, w = x.size(2), x.size(3)
        # f = self.feats(x)[-1]
        x1, x2, x3, x4 = self.feats(x) # size : 128, 64, 32, 16
        x1 = self.down_1(x1)
        x2 = self.down_2(x2)
        x3 = self.down_3(x3)
        x4 = self.down_4(x4)
        h0, w0 = x2.shape[-2:]
        x1 = F.interpolate(x1, (h0, w0), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, (h0, w0), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, (h0, w0), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, (h0, w0), mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.down_5(x)

        p = self.psp(x)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        logit = self.final(p)
        logit = F.upsample(input=logit, size=(h, w), mode='bilinear')
        if self.training:
            return self.loss(logit, y['cls'])

        else:
            return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=False,
                freeze_at=0,
                # 16 or 32
                output_stride=8,
                with_cp=(False, False, False, False),
                norm_layer=nn.BatchNorm2d,
            ),
            classes=7,
            loss=dict(
                ignore_index=-1,
                ce=dict()
            ),
            psp=dict(
                sizes=(1, 2, 3, 6),
                psp_size=2048,
                deep_features_size=1024
            )
        ))

if __name__ == '__main__':
    m = PSPNet_DensePPM(dict())
    m.eval()
    o = m(torch.ones(2, 3, 512, 512))
    print(o.shape)