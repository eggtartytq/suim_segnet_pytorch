"""
Train a SegNet model
Usage:
python train.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                --train_path ImageSets/Segmentation/train.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir /home/SharedData/intern_sayan/PascalVOC2012/ \
                --checkpoint /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                --gpu 1
"""

from __future__ import print_function
import argparse
from getdata import PascalVOCDataset, NUM_CLASSES
from segnet import SegNet
import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision


# Constants

NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
MOMENTUM = 0.9
BATCH_SIZE = 32

transform = torchvision.transforms.Compose([
torchvision.transforms.ToTensor(), 
torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                    ])

# Arguments
# parser = argparse.ArgumentParser(description='Train a SegNet model')

# parser.add_argument('--data_root', required=True)
# parser.add_argument('--train_path', required=True)
# parser.add_argument('--img_dir', required=True)
# parser.add_argument('--mask_dir', required=True)
# parser.add_argument('--save_dir', required=True)
# parser.add_argument('--checkpoint')
# parser.add_argument('--gpu', type=int)

# args = parser.parse_args()


def train():
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()

        for batch in train_dataloader:
            input_tensor = torch.autograd.Variable(batch['image']).permute(0, 3, 1, 2)
            target_tensor = torch.autograd.Variable(batch['mask']).permute(0, 3, 1, 2).type(torch.float32)
            if CUDA:
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()


            # print(input_tensor.shape)

            # predicted_tensor, sigmoid_tensor = model(input_tensor)
            predicted_tensor, softmax_tensor = model(input_tensor)


            #import ipdb; ipdb.set_trace()


            optimizer.zero_grad()
            # loss = criterion(predicted_tensor, target_tensor)
            loss = criterion(predicted_tensor, target_tensor)
            loss.backward()
            optimizer.step()


            loss_f += loss.float()
            # prediction_f = sigmoid_tensor.float()

        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best_transform_2.pth"))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))


if __name__ == "__main__":
    save_dir = './save_model'
    img_dir = './train_val/images'
    mask_dir = './train_val/masks'
    checkpoint = os.path.join(save_dir, "model_best_transform_2.pth")

    CUDA = 1
    


    train_dataset = PascalVOCDataset(img_dir=img_dir,
                                     mask_dir=mask_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)

    USE_CUDA = True
    if USE_CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda()
        weight = torch.ones(NUM_CLASSES)
        
        criterion = torch.nn.CrossEntropyLoss(weight).cuda()
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)
        weight = torch.ones(NUM_CLASSES)
        
        criterion = torch.nn.CrossEntropyLoss(weight)


    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))


    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)


    # optimizer = torch.optim.Adam(model.encoder.parameters(), model.decoder.parameters(),
    #                                  lr=LEARNING_RATE)


    train()