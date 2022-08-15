import numpy as np
from PIL import Image

def colorful(img,palette):
    #img:需要上色的图片(class PIL.Image)
    img = Image.fromarray(np.uint8(img))
    img.putpalette(palette)
    return img