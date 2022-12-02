import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils.VisSeg import VisSeg
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)
PALETTE = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()



def predict_test(pseudo_label_path):
    pseudo_label_name = pseudo_label_path.split('\\')[-1]
    pseudo_label_vis_dir = (pseudo_label_path.split('\\')[0:-2])
    pseudo_label_vis_dir.append('vis')
    pseudo_label_vis_dir = os.path.join(*pseudo_label_vis_dir)
    viz_op = VisSeg(PALETTE, pseudo_label_vis_dir)

    pseudo_label = Image.open(pseudo_label_path)
    pseudo_label = np.array(pseudo_label) -1
    # pseudo_label的取值范围为[0, 255]   ==>   0-1 = 255
    pseudo_label[pseudo_label == 255] = 0
    viz_op.saveVis(pseudo_label, pseudo_label_name)



if __name__ == '__main__':
    file_path = r"wandb\Ablation_Study_ICS\No_ICS\files\pseudo_label"
    for step in range(4000, 20000, 1000):
        print("vis step "+ str(step) + " :")
        png_path = os.path.join(file_path, str(step), 'pred')
        png_name_list = os.listdir(png_path)
        for png_name in tqdm(png_name_list):
            pseudo_label_path = os.path.join(png_path, png_name)
            predict_test(pseudo_label_path)