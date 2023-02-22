from torch import nn
import torch
import torch.nn.functional as F

class DensePPM(nn.Module):
    def __init__(self, in_channels,reduction_dim, pool_sizes, norm_layer = nn.BatchNorm2d):
        super(DensePPM, self).__init__()
        n=1
        self.out_channels = in_channels + reduction_dim*((len(pool_sizes)*len(pool_sizes) + len(pool_sizes))//2)
        self.stages =[]
        for pool_size in pool_sizes:
            stage=self._make_stages(in_channels+reduction_dim*((n*n-n)//2), reduction_dim*n, pool_size, norm_layer)
            self.stages.append(stage)
            n+=1
        self.stages=nn.ModuleList(self.stages)   

    def _make_stages(self, in_channels, reduction_dim, bin_sz, norm_layer):
        prior = nn.AvgPool2d(kernel_size = bin_sz, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        conv = nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=False)
        bn = norm_layer(reduction_dim)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        for stage in self.stages:
            out=stage(features)
            out=F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            features=torch.cat([features,out], dim=1)
        return features