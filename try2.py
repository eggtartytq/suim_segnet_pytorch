import torch
import os
import torchvision.transforms as T
from PIL import Image

img_dir = './test_val/images'
mask_dir = './test_val/masks'

image = Image.open(img_dir)
mask = mask.open(mask_dir)

both_images = torch.cat((image.unsqueeze(0), mask.unsqueeze(0)),0)

transformed_images = T.RandomRotation(180)(both_images)

image_trans = transformed_images[0]
mask_trans = transformed_images[1]

image_trans.save('test_image1.jpg')
mask_trans.save('test_mask.bmp')