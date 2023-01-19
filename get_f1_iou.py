"""
# Script for evaluating F score and mIOU 
"""
from __future__ import print_function, division
import ntpath
import numpy as np
from PIL import Image
# local libs
from score import getPaths,db_eval_boundary, IoU_bin
import pandas as pd

obj_list = ["HD/", "WR/", "RO/", "RI/", "FV/"]
# dataset_list = ['Original/', 'Brightness/', 'Contrast/', 'Defocus_Blur/', 'Elastic/', 'Gaussian_Noise/', 'Impulse_Noise/'
#                 , 'jpeg_comp/', 'Motion_Blur/', 'Pixelate/', 'Shot_Noise/', 'Zoom_Blur/']
## experiment directories
#obj_cat = "RI/" # sub-dir  ["RI/", "FV/", "WR/", "RO/", "HD/"]
test_dir = "test_val/masks/"
test_mask_dir = "test_val/output/segnet/"
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/masks/"
#real_mask_dir = test_dir + obj_cat # real labels
#gen_mask_dir = "test_val/output/unet/Original/" + obj_cat # generated labels

## input/output shapes
im_res = (256, 320) 

# for reading and scaling input images
def read_and_bin(im_path):
    img = Image.open(im_path).resize(im_res)
    img = np.array(img)/255.
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

# accumulate F1/iou values in the lists
def cal_score(gen_mask_dir, real_mask_dir):
    Ps, Rs, F1s, IoUs = [], [], [], []
    gen_paths = sorted(getPaths(gen_mask_dir))
    real_paths = sorted(getPaths(real_mask_dir))
    for gen_p, real_p in zip(gen_paths, real_paths):
        gen, real = read_and_bin(gen_p), read_and_bin(real_p)
        if (np.sum(real)>0):
            precision, recall, F1 = db_eval_boundary(real, gen)
            iou = IoU_bin(real, gen)
            #print ("{0}:>> P: {1}, R: {2}, F1: {3}, IoU: {4}".format(gen_p, precision, recall, F1, iou))
            Ps.append(precision) 
            Rs.append(recall)
            F1s.append(F1)
            IoUs.append(iou)

    # print F-score and mIOU in [0, 100] scale
    f_score = format(100.0*np.mean(F1s),'.2f')
    iou_score = format(100.0*np.mean(IoUs), '.2f')
    print ("Avg. F: {0}".format(100.0*np.mean(F1s)))
    print ("Avg. IoU: {0}".format(100.0*np.mean(IoUs)))
    return f_score,iou_score
    



#Get score
f_scores = pd.DataFrame(
        columns=obj_list,)
iou_scores = pd.DataFrame(
        columns=obj_list,)
# for dataset in dataset_list:
#     print('###############################')
#     print(dataset)
#     mask_dir = test_mask_dir + dataset
#     for obj in obj_list:
#         print('---------------------------')
#         print(obj)
#         real_mask_dir = test_dir + obj
#         gen_mask_dir = mask_dir + obj
#         temp_f, temp_iou = cal_score(gen_mask_dir, real_mask_dir)
#         f_scores.loc[dataset,obj] = temp_f
#         iou_scores.loc[dataset,obj] = temp_iou

root_real = './test_val/masks/'
root_gen = './output/'
for obj in obj_list:
    real_mask_dir = root_real + obj
    gen_mask_dir = root_gen + obj
    temp_f, temp_iou = cal_score(gen_mask_dir, real_mask_dir)
    f_scores.loc[obj] = temp_f
    iou_scores.loc[obj] = temp_iou

print(f_scores)
f_scores.to_csv("segnet_retrain0_f.csv")
iou_scores.to_csv("segnet_retrain0_iou.csv")