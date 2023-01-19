from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

#shape1: h: 256, w: 320
#shape2: h:224, w:224

SUIM_CLASSES = ('background',
                'Human',
               'Robot',
               'Fish',
               'Reef',
               'Wreck')

NUM_CLASSES = len(SUIM_CLASSES) 



class PascalVOCDataset(Dataset):
    
    def __init__(self, img_dir, mask_dir, image_shape=(224, 224), n_channels=3, transform=None):
        #self.images = open(list_file, "rt").read().split("\n")[:-1]
        
        self.imgshape = image_shape
        self.n_channels = n_channels
        self.img_extension = ".jpg"
        self.mask_extension = ".bmp"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir
        self.transform = transform

        #self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(os.listdir(self.image_root_dir))

    def __getitem__(self, index):
        image_name = os.listdir(self.image_root_dir)[index]
        name = image_name.split('.')[0]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }

        return data

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        raw_image = transform.Grayscale(3)(raw_image)
        imx_t = np.array(raw_image, dtype=np.float32)/255.0
        
        # print(imx_t.shape)
        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        raw_image = np.array(raw_image)
        raw_image = raw_image/255.0
        raw_image[raw_image > 0.5] = 1
        raw_image[raw_image <= 0.5] = 0
        mask = []
        mask.append(self.getDiffMask(raw_image))
        # import ipdb;ipdb.set_trace()
        imx_t = np.array(mask).squeeze()
        # print(imx_t.shape)
        # border
        #imx_t[imx_t==255] = len(VOC_CLASSES)

        return imx_t

    def getDiffMask(self, mask):
        imw, imh = mask.shape[0], mask.shape[1]

        
        #################
        # softmax
        #################
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
        
        '''
        #################
        # Sigmoid
        #################
        Human = np.zeros((imw, imh))
        Robot = np.zeros((imw, imh))
        Fish = np.zeros((imw, imh))
        Reef = np.zeros((imw, imh))
        Wreck = np.zeros((imw, imh))

        mask_idx = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
        Human[mask_idx] = 1
            
        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 0))
        Robot[mask_idx] = 1

        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 1) & (mask[:,:,2] == 0))
        Fish[mask_idx] = 1

        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
        Reef[mask_idx] = 1

        mask_idx = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 1) & (mask[:,:,2] == 1))
        Wreck[mask_idx] = 1

        return np.stack(( Robot, Fish, Human, Reef, Wreck), -1) 
        '''
if __name__ == "__main__":
    img_dir = './train_val/images'
    mask_dir = './train_val/masks'


    objects_dataset = PascalVOCDataset(img_dir=img_dir,
                                       mask_dir=mask_dir)

    #print(objects_dataset.get_class_probability())

    sample = objects_dataset[519]
    image, mask = sample['image'], sample['mask']

    #image.transpose_(0, 2)

    fig = plt.figure()

    a = fig.add_subplot(1,2,1)
    plt.imshow(image)
    print(image.shape)
    a = fig.add_subplot(1,2,2)
    print(mask.shape)
    mask.transpose_(0,2)
    mask.transpose_(0,1)
    plt.imshow(mask[:,:,:, 0])
    #fig.savefig('test_0.png')
    plt.show()

    # 0: robot
    # 1: fish
    # 2: human
    # 3: reef
    # 4: wreck
    