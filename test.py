"""Test for SegNet"""

from __future__ import print_function
from segnet import SegNet
from getdata import PascalVOCDataset, NUM_CLASSES
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from os.path import join, exists
from torch import nn,optim
from PIL import Image
import numpy as np
import torch
import ntpath

BATCH_SIZE = 4

if __name__ == "__main__":
    # RGB input
    input_channels = 3
    # RGB output
    output_channels = NUM_CLASSES

    im_h = 224
    im_w = 224
    # #test files
    # root_dir = './test_val'
    # test_dir = root_dir + '/images/'
    # RO_dir = root_dir + "/RO/"
    # FB_dir = root_dir + "/FV/"
    # WR_dir = root_dir + "/WR/"
    # HD_dir = root_dir + "/HD/"
    # RI_dir = root_dir + "/RI/" 
    #output files
    samples_dir = './output/transform_2'
    RO_dir = samples_dir + "/RO/"
    FB_dir = samples_dir + "/FV/"
    WR_dir = samples_dir + "/WR/"
    HD_dir = samples_dir + "/HD/"
    RI_dir = samples_dir + "/RI/"
    if not exists(samples_dir): os.makedirs(samples_dir)
    if not exists(RO_dir): os.makedirs(RO_dir)
    if not exists(FB_dir): os.makedirs(FB_dir)
    if not exists(WR_dir): os.makedirs(WR_dir)
    if not exists(HD_dir): os.makedirs(HD_dir)
    if not exists(RI_dir): os.makedirs(RI_dir)

    # Model
    model = SegNet(input_channels=input_channels, output_channels=output_channels)
    print(model)
    state_dict = torch.load('./save_model/model_best_transform_2.pth')
    model.load_state_dict(state_dict)
    
    img_dir = './test_val/images'
    mask_dir = './test_val/masks/whole'


    test_dataset = PascalVOCDataset(img_dir=img_dir,
                                       mask_dir=mask_dir)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=4)
    count = 0
    name_lst = os.listdir(mask_dir)
    for batch in test_dataloader:
        input_tensor = torch.autograd.Variable(batch['image']).permute(0, 3, 1, 2)
        
        target_tensor = torch.autograd.Variable(batch['mask']).permute(0, 3, 1, 2).type(torch.float32)
        predicted_tensor, sigmoid_tensor = model(input_tensor)
        #import ipdb; ipdb.set_trace()
        # predicted_tensor, softmaxed_tensor = model(input_tensor)
        # softmaxvalues = softmaxed_tensor.argmax(axis=1).detach().numpy()
        
        #print(predicted_tensor.shape)
        for i in range(sigmoid_tensor.shape[0]):
            out_img = sigmoid_tensor[i,:,:,:]
            out_img[out_img>0.5] = 1.
            out_img[out_img<=0.5] = 0.

            #import ipdb; ipdb.set_trace()
            #print(out_img.shape)
            
            img_name = name_lst[count].split('.')[0]+'.bmp'
            count += 1

            '''
            #############
            # Softmax
            #############
            Human = np.zeros((im_h, im_w))
            Robot = np.zeros((im_h, im_w))
            Fish = np.zeros((im_h, im_w))
            Reef = np.zeros((im_h, im_w))
            Wreck = np.zeros((im_h, im_w))

            BG = np.where(softmaxvalues[i] == 0)
            ROs = np.where(softmaxvalues[i]  == 1)
            FVs = np.where(softmaxvalues[i]  == 2)
            HDs = np.where(softmaxvalues[i]  == 3)
            RIs = np.where(softmaxvalues[i]  == 4)
            WRs = np.where(softmaxvalues[i]  == 5)

            Human[HDs] = 1
            Robot[ROs] = 1
            Fish[FVs] = 1
            Reef[RIs] = 1
            Wreck[WRs] = 1
            '''



            ROs = np.reshape(out_img[0,:,:].detach().numpy(), (im_h, im_w))
            FVs = np.reshape(out_img[1,:,:].detach().numpy(), (im_h, im_w))
            HDs = np.reshape(out_img[2,:,:].detach().numpy(), (im_h, im_w))
            RIs = np.reshape(out_img[3,:,:].detach().numpy(), (im_h, im_w))
            WRs = np.reshape(out_img[4,:,:].detach().numpy(), (im_h, im_w))
            Image.fromarray(np.uint8(ROs*255.)).save(RO_dir+img_name)
            Image.fromarray(np.uint8(FVs*255.)).save(FB_dir+img_name)
            Image.fromarray(np.uint8(HDs*255.)).save(HD_dir+img_name)
            Image.fromarray(np.uint8(RIs*255.)).save(RI_dir+img_name)
            Image.fromarray(np.uint8(WRs*255.)).save(WR_dir+img_name)

            '''
            #################
            #Softmax
            #################
            Image.fromarray(np.uint8(Robot*255.)).save(RO_dir+img_name)
            Image.fromarray(np.uint8(Fish*255.)).save(FB_dir+img_name)
            Image.fromarray(np.uint8(Human*255.)).save(HD_dir+img_name)
            Image.fromarray(np.uint8(Reef*255.)).save(RI_dir+img_name)
            Image.fromarray(np.uint8(Wreck*255.)).save(WR_dir+img_name)
            '''
            print('done image: {}', i)
    '''
    imgs = []
    for p in getPaths(test_dir):
        img = Image.open(p).resize((256,320))
        img = np.array(img)/255.
        img = np.expand_dims(img, axis=0)
        predicted_out = model(img)
    '''