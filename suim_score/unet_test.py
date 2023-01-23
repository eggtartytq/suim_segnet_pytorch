"""
# Test script for the UNet
    # for 5 object categories: HD, FV, RO, RI, WR 
# See https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
# local libs
from models.unet import UNet0
from utils.data_utils import getPaths

import matplotlib.pyplot as plt
from matplotlib import cm as col_map
import seaborn as sns
from scipy.stats import entropy
from scipy.special import softmax
from sklearn.metrics import roc_auc_score



## experiment directories
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/images/"
test_dir = "./test_val/images/"
test_dir_mask = './test_val/masks/'

test_dir_out = "./test_val/severity_lv5/Defocus_Blur/"
## sample and ckpt dir
samples_dir = "./test_val/output/unet_defocus_lv5/"
RO_dir = samples_dir + "RO/"
FB_dir = samples_dir + "FV/"
WR_dir = samples_dir + "WR/"
HD_dir = samples_dir + "HD/"
RI_dir = samples_dir + "RI/" 
if not exists(samples_dir): os.makedirs(samples_dir)
if not exists(RO_dir): os.makedirs(RO_dir)
if not exists(FB_dir): os.makedirs(FB_dir)
if not exists(WR_dir): os.makedirs(WR_dir)
if not exists(HD_dir): os.makedirs(HD_dir)
if not exists(RI_dir): os.makedirs(RI_dir)

## input/output shapes
im_res_ = (320, 240, 3) 
ckpt_name = "unet_rgb5.hdf5"
model = UNet0(input_size=(im_res_[1], im_res_[0], 3), no_of_class=5)
print (model.summary())
model.load_weights(join("ckpt/", ckpt_name))


im_h, im_w = im_res_[1], im_res_[0]
def testGenerator():
    # test all images in the directory
    assert exists(test_dir), "local image path doesnt exist"
    imgs = []
    ###########
    #score list
    ###########
    data_ent = []
    data_auc = []
    for p in getPaths(test_dir):
        mask_path = test_dir_mask +  p.split('/')[-1][:-4] + '.bmp'
        image_name = p.split('/')[-1][:-4] + '.jpg'

        # read and scale inputs
        image_ = Image.open(p).resize((im_w, im_h))
        img = np.array(image_)/255.
        img = np.expand_dims(img, axis=0)

        # read and scale inputs
        out_path = test_dir_out + image_name[:-4] + 'defocus_.jpg'
        image_out = Image.open(out_path).resize((im_w, im_h))
        img_out = np.array(image_out)/255.
        img_out = np.expand_dims(img_out, axis=0)


        

        # read mask
        mask = gt_mask_loader(mask_path, (im_w, im_h))

        # inference
        ori_img_out = model.predict(img).squeeze()
        out_img_out = model.predict(img_out).squeeze()

        # Uncertainty Evaluation

        in_ent = entropy(softmax(ori_img_out, axis=-1), axis= -1)
        out_ent = entropy(softmax(out_img_out, axis=-1), axis= -1)

        ent_auc = auc_score(in_ent, out_ent)
        data_ent.append(ent_auc)
        print(ent_auc)


        in_var = np.var(softmax(ori_img_out, axis=-1), axis= -1)
        out_var = np.var(softmax(out_img_out, axis=-1), axis= -1)

        var_auc = auc_score(in_var, out_var)
        data_auc.append(var_auc)
        print(var_auc)

        


        # thresholding
        ori_img_out[ori_img_out>0.5] = 1.
        ori_img_out[ori_img_out<=0.5] = 0.


        out_img_out[out_img_out>0.5] = 1.
        out_img_out[out_img_out<=0.5] = 0.



        # import ipdb; ipdb.set_trace()


        out_class_labels = ori_img_out.argmax(axis=-1)
        incorrectPrediction = np.zeros_like(out_class_labels)
        incorrectPrediction[np.where(mask[:,:,1:].argmax(axis=-1) != out_class_labels)] = 255
        
        out_class_labels = out_img_out.argmax(axis=-1)
        incorrectPrediction_out_d = np.zeros_like(out_class_labels)
        incorrectPrediction_out_d[np.where(mask[:,:,1:].argmax(axis=-1) != out_class_labels)] = 255
        

        print ("tested: {0}".format(p))
        # get filename
        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        # save individual output masks
        ROs = np.reshape(ori_img_out[:,:,0], (im_h, im_w))
        FVs = np.reshape(ori_img_out[:,:,1], (im_h, im_w))
        HDs = np.reshape(ori_img_out[:,:,2], (im_h, im_w))
        RIs = np.reshape(ori_img_out[:,:,3], (im_h, im_w))
        WRs = np.reshape(ori_img_out[:,:,4], (im_h, im_w))
        Image.fromarray(np.uint8(ROs*255.)).save(RO_dir+img_name)
        Image.fromarray(np.uint8(FVs*255.)).save(FB_dir+img_name)
        Image.fromarray(np.uint8(HDs*255.)).save(HD_dir+img_name)
        Image.fromarray(np.uint8(RIs*255.)).save(RI_dir+img_name)
        Image.fromarray(np.uint8(WRs*255.)).save(WR_dir+img_name)



        visualize_gt_pred(image_, mask, ori_img_out, in_ent, incorrectPrediction, image_name)
    
    #calculation
    ent_lst = np.array(data_ent)
    auc_lst = np.array(data_auc)

    ent_mean = np.average(ent_lst)
    auc_mean = np.average(auc_lst)
    print('*************************************')
    print('ent score is: {}', ent_mean)
    print('var score is: {}', auc_mean)
    print('*************************************')


def auc_score(inliers, outliers):
    """Computes the AUROC score w.r.t network outputs on two distinct datasets.
    Typically, one dataset is the main training/testing set, while the
    second dataset represents a set of unseen outliers.
    
    Args: 
        inliers (torch.tensor): set of predictions on inlier data
        outliers (torch.tensor): set of predictions on outlier data
    
    Returns:
        AUROC score (float)
    """
    # inliers = inliers.detach().cpu().numpy()
    # outliers = outliers.detach().cpu().numpy()
    inliers = inliers.reshape(-1)
    outliers = outliers.reshape(-1)
    # import ipdb; ipdb.set_trace()

    y_true = np.array([0] * len(inliers) + [1] * len(outliers))
    y_score = np.concatenate([inliers, outliers])
    try:
        auc_score = roc_auc_score(y_true, y_score)
    except NameError:
        absl.logging.info('roc_auc_score function not defined')
        auc_score = 0.5
    return auc_score

def gt_mask_loader(path, mask_size):
        mask = Image.open(path)
        mask = mask.resize((mask_size[0], mask_size[1]))
        mask = np.asarray(mask)
        mask = mask/mask.max()
        imw, imh = mask.shape[0], mask.shape[1]   
       
        background = np.ones((imw, imh))
        Human1 = np.zeros((imw, imh))
        Robot1 = np.zeros((imw, imh))
        Fish1 = np.zeros((imw, imh))
        Reef1 = np.zeros((imw, imh))
        Wreck1 = np.zeros((imw, imh))

        mask_idx = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
        Human1[mask_idx] = 1
        background[mask_idx] = 0
    
        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 0))
        Robot1[mask_idx] = 1
        background[mask_idx] = 0

        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 1) & (mask[:,:,2] == 0))
        Fish1[mask_idx] = 1
        background[mask_idx] = 0

        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
        Reef1[mask_idx] = 1
        background[mask_idx] = 0

        mask_idx = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 1) & (mask[:,:,2] == 1))
        Wreck1[mask_idx] = 1
        background[mask_idx] = 0
        return np.stack((background, Robot1, Fish1, Human1, Reef1, Wreck1), -1) 
        


def visualize_gt_pred(images, gt_map, pred_map, entroy_map, incorrectPrediction, image_name):
    label_to_color = {
    0: [0,  0, 0],
    1: [0,  0, 255],
    2: [ 0,  255,  255],
    3: [255, 0, 0],
    4: [255, 20, 147],
    5: [255, 255, 0],
    
}

    label_to_color2 = {
    1: [0,  0, 255],
    2: [ 0,  255,  255],
    3: [255, 0, 0],
    4: [255, 20, 147],
    5: [255, 255, 0],
    
}



    f, axarr = plt.subplots(1,5, figsize=(10, 10))
    img_sample = images
    gt_sample = gt_map.argmax(axis=-1).squeeze()
    pred_sample = pred_map

    img_rgb_pred = np.zeros((pred_sample.shape[0], pred_sample.shape[1], 3), dtype=np.uint8)
    img_rgb_gt = np.zeros((pred_sample.shape[0], pred_sample.shape[1], 3), dtype=np.uint8)
    for gray, rgb in label_to_color.items():
        img_rgb_gt[gt_sample == gray, :] = rgb

    for gray, rgb in label_to_color2.items():
        img_rgb_pred[pred_sample[:,:,gray-1] == 1, :] = rgb

    axarr[0].imshow(img_sample)    
    axarr[1].imshow(img_rgb_gt)
    axarr[2].imshow(img_rgb_pred)
    axarr[3].imshow(incorrectPrediction, cmap=plt.cm.gray)
    axarr[4].imshow(entroy_map, cmap = col_map.jet)


    for axx in axarr:
        axx.set_xticks([])
        axx.set_yticks([])


    uncertainity_dir = './Un_/'
    if not exists(uncertainity_dir): os.makedirs(uncertainity_dir)
    save_path = uncertainity_dir
    name = image_name
    
    for axx in axarr:
        axx.set_xticks([])
        axx.set_yticks([])

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(save_path + name)
    




# test images
testGenerator()


