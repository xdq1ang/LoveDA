from matplotlib import pyplot as plt
import os
from utils.colorful import colorful
from torchvision import transforms
import torch


class VisSeg(object):
    def __init__(self, palette, savepath):
        self.palette = palette
        self.savePath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)

    def saveVis(self, mask, filename):
        mask = colorful(mask, self.palette)
        mask.save(os.path.join(self.savePath, filename))
        return mask
