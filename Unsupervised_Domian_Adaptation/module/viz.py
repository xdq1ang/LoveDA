from PIL import Image
import numpy as np
import os
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

class VisualizeSegmm(object):
    def __init__(self, out_dir, palette):
        self.out_dir = out_dir
        self.palette = palette
        os.makedirs(self.out_dir, exist_ok=True)

    def setpalette(self, y_pred, filename, save=True):
        """
        Args:
            y_pred: 2-D or 3-D array of shape [1 (optional), H, W]
            filename: str
        Returns:
        """
        y_pred = y_pred.astype(np.uint8)
        y_pred = y_pred.squeeze()
        color_y = Image.fromarray(y_pred)
        color_y.putpalette(self.palette)
        if (save):
            color_y.save(os.path.join(self.out_dir, filename))
        return color_y

    def saveheatmap(self, domain_dis, filename):
        # img = Image.fromarray(domain_dis)
        # img = img.convert('RGB')
        # img.save(os.path.join(self.out_dir, filename))
        fig = plt.figure()
        sns_plot = sns.heatmap(domain_dis)
        fig.savefig(os.path.join(self.out_dir, filename), bbox_inches='tight', pad_inches=0)
        plt.close()