import segmentation_models_pytorch as smp
import ever as er
from module.loss import SegmentationLoss
from module.baseline.denseppm import DensePPM
from torch import nn


class Downsample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Downsample, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.Conv_BN_ReLU_2(x)
        return x

@er.registry.MODEL.register()
class DeepLabV3_DensePPM(er.ERModule, ):
    def __init__(self, config):
        super(DeepLabV3_DensePPM, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.encoder = smp.DeepLabV3(self.config.encoder_name,
                                    encoder_weights=self.config.encoder_weights,
                                    classes=self.config.classes,
                                    activation=None
                                    ).encoder
        
        self.decoder = DensePPM(self.encoder.out_channels[-1], 64, [2, 3, 4, 5])
        self.segmentation_head = nn.Conv2d(self.decoder.out_channels, config.classes, kernel_size=3, padding=3 // 2)
        

    def forward(self, x, y=None):
        b, c, w, h = x.shape
        feature = self.encoder(x)[-1]

        feature = self.decoder(feature)
        logit = self.segmentation_head(feature)
        logit = nn.UpsamplingBilinear2d(size=(w, h))(logit)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))