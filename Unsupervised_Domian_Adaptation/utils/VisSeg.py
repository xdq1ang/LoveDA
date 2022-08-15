from matplotlib import pyplot as plt
import os
from utils.colorful import colorful
from torchvision import transforms
import torch


class VisSeg(object):
    def __init__(self, palette, savePath):
        self.palette = palette
        self.savePath = savePath
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    def Vis(self, mask, fileName):
        mask = colorful(mask,self.palette)
        mask.save(os.path.join(self.savePath,fileName))

    