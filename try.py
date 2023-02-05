import albumentations as A
import os
import matplotlib.pyplot as plt

from PIL import Image

current_file=os.path.dirname(os.path.abspath(__file__))
img_dir = 'D:\Code\Github\suim_segnet_pytorch\try_val\images\d_r_1_.jpg'
mask_dir = 'D:\Code\Github\suim_segnet_pytorch\try_val\masks\d_r_1_.bmp'

image = Image.open(img_dir)
mask = Image.open(mask_dir)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    
    A.RandomCrop(height=224, width=224, p=0.5), 
], p=1)

transformed = transform(image=image, mask=mask)

image_trans = transformed['image']
mask_trans = transformed['mask']

image_trans.save("test_image0.jpg")
mask_trans.save("test_mask0.bmp")