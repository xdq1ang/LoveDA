from torch import nn
import torch
import torch.nn.functional as F


class DensePPM(nn.Module):
    def __init__(self, in_channels, num_classes, reduction_dim, pool_sizes, norm_layer=nn.BatchNorm2d):
        super(DensePPM, self).__init__()
        n = 1
        self.stages = []
        for pool_size in pool_sizes:
            # stage=self._make_stages(in_channels+reduction_dim*n, reduction_dim, pool_size, norm_layer)
            stage = self._make_stages(in_channels + reduction_dim * ((n * n - n) // 2), reduction_dim * n, pool_size,
                                      norm_layer)
            self.stages.append(stage)
            n += 1
        self.stages = nn.ModuleList(self.stages)
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels + reduction_dim * ((len(pool_sizes) * len(pool_sizes) + len(pool_sizes)) // 2), 512,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def _make_stages(self, in_channels, reduction_dim, bin_sz, norm_layer):
        # prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        prior = nn.AvgPool2d(kernel_size=bin_sz, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        conv = nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=False)
        bn = norm_layer(reduction_dim)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        for stage in self.stages:
            out = stage(features)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            features = torch.cat([features, out], dim=1)
        out = self.conv_last(features)
        return out
