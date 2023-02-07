import segmentation_models_pytorch as smp
import ever as er
from module.loss import SegmentationLoss
from module.baseline.denseppm import DensePPM


@er.registry.MODEL.register()
class AnyUNet_DensePPM(er.ERModule, ):
    def __init__(self, config):
        super(AnyUNet_DensePPM, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.Unet(self.config.encoder_name,
                                 encoder_weights=self.config.encoder_weights,
                                 classes=self.config.classes,
                                 activation=None
                                 )
        # 此处需要修改
        self.ppmLayer = DensePPM(in_channels = 2048, reduction_dim = 64, pool_sizes = [2, 3, 4, 5])

    def forward(self, x, y=None):
        logit = self.features(x)
        
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
