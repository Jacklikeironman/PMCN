import imageio
from torch.utils.data.dataset import Dataset
import torch
from torch.utils.data import DataLoader
import random
import torch
import torch.nn.functional as F
import os
import numpy as np

def random_crop_and_pad_image_and_labels(images, size):
    image_shape = images.shape
    combined_pad = F.pad(images, (0, max(size[1], image_shape[3]) - image_shape[3], 0, max(size[0], image_shape[2]) - image_shape[2]))
    freesize0 = random.randint(0, max(size[0], image_shape[2]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[3]) - size[1])
    combined_crop = combined_pad[:, :, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return combined_crop

def random_flip(combined_crop):  
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1
    if transforms and vertical_flip and random.randint(0, 1) == 1:
        combined_crop = torch.flip(combined_crop, [2])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        combined_crop = torch.flip(combined_crop, [3])
    return combined_crop

class Trainval_Dataset(Dataset):
    ###############################
    # Input:
    #   patch_size (int): the crop size of input patches
    #   mode (str): train or val
    # Return:
    #   label_left_patches, label_right_patches, fog_left_patches, fog_right_patches
    #   shape = [B, 7, C, H (patch_size), W (patch_size)]
    ###############################
    def __init__(self, patch_size, fog_intensity, mode="train"):
        self.im_height = patch_size
        self.im_width = patch_size
        self.fog_intensity = fog_intensity
        self.mode = mode
        self.head_frame_index_list, self.image_lists = self.get_image()

        if self.mode == 'train':
            print("Train dataset find sequences: ", len(self.head_frame_index_list))
        else:
            print("Val dataset find sequences: ", len(self.head_frame_index_list))


    def get_image(self):
        rootdir='Cityscapes/Stereo_Foggy_Video_Cityscapes'
        label_left_list='Cityscapes/lists_file_names_{}/leftImg8bit_trainval_refined_clean_filenames.txt'.format(str(self.fog_intensity))
        label_right_list='Cityscapes/lists_file_names_{}/rightImg8bit_trainval_refined_filenames.txt'.format(str(self.fog_intensity))
        fog_left_list='Cityscapes/lists_file_names_{}/leftImg8bit_trainval_fog_filenames.txt'.format(str(self.fog_intensity))
        fog_right_list='Cityscapes/lists_file_names_{}/rightImg8bit_trainval_fog_filenames.txt'.format(str(self.fog_intensity))

        with open(label_left_list) as f1:
            label_left_data = f1.readlines()
        with open(label_right_list) as f2:
            label_right_data = f2.readlines()
        with open(fog_left_list) as f3:
            fog_left_data = f3.readlines()
        with open(fog_right_list) as f4:
            fog_right_data = f4.readlines()
        
        image_lists = {'label_left':[], "label_right":[], "fog_left":[], "fog_right":[]}
        for _, line in enumerate(label_left_data, 1):
            if self.mode in line:
                image_lists['label_left'] += [os.path.join(rootdir, line.rstrip())]
        for _, line in enumerate(label_right_data, 1):
            if self.mode in line:
                image_lists['label_right'] += [os.path.join(rootdir, line.rstrip())]
        for _, line in enumerate(fog_left_data, 1):
            if self.mode in line:
                image_lists['fog_left'] += [os.path.join(rootdir, line.rstrip())]
        for _, line in enumerate(fog_right_data, 1):
            if self.mode in line:
                image_lists['fog_right'] += [os.path.join(rootdir, line.rstrip())]

        # check lists
        assert len(image_lists['label_left']) == len(image_lists['label_right']) == len(image_lists['fog_left']) == len(image_lists['fog_right'])
        for i in range(len(image_lists['label_left'])):
            info = image_lists['label_left'][i].split('/')[-1].split('_')[0] + '_' + image_lists['label_left'][i].split('/')[-1].split('_')[1] + '_' + image_lists['label_left'][i].split('/')[-1].split('_')[2]
            assert info in image_lists['label_right'][i]
            assert info in image_lists['fog_left'][i]
            assert info in image_lists['fog_right'][i]

        head_frame_index_list = list(np.linspace(0, len(image_lists['label_left']), len(image_lists['label_left']) // 7, endpoint=False, dtype=np.int))
        for h_idx in head_frame_index_list:
            for i in range(1, 7):
                info = image_lists['label_left'][h_idx].split('/')[-1].split('_')[0] + '_' + image_lists['label_left'][h_idx].split('/')[-1].split('_')[1]
                assert info in image_lists['label_left'][h_idx + i]
                assert info in image_lists['label_right'][h_idx + i]
                assert info in image_lists['fog_left'][h_idx + i]
                assert info in image_lists['fog_right'][h_idx + i]
 
        return head_frame_index_list, image_lists

    def __len__(self):
        return len(self.head_frame_index_list)
    
    def __getitem__(self, index):
        head_frame_index = self.head_frame_index_list[index]

        label_left_images = []
        label_right_images = []
        fog_left_images = []
        fog_right_images = []

        for n in range(7):
            label_left_images.append((imageio.imread(self.image_lists['label_left'][head_frame_index + n]).transpose(2, 0, 1)).astype(np.float32) / 255.0)
            label_right_images.append((imageio.imread(self.image_lists['label_right'][head_frame_index + n]).transpose(2, 0, 1)).astype(np.float32) / 255.0)
            fog_left_images.append((imageio.imread(self.image_lists['fog_left'][head_frame_index + n]).transpose(2, 0, 1)).astype(np.float32) / 255.0)
            fog_right_images.append((imageio.imread(self.image_lists['fog_right'][head_frame_index + n]).transpose(2, 0, 1)).astype(np.float32) / 255.0)

        label_left_images = np.array(label_left_images)
        label_right_images = np.array(label_right_images)
        fog_left_images = np.array(fog_left_images)
        fog_right_images = np.array(fog_right_images)

        label_left_images = torch.from_numpy(label_left_images).float()
        label_right_images = torch.from_numpy(label_right_images).float()
        fog_left_images = torch.from_numpy(fog_left_images).float()
        fog_right_images = torch.from_numpy(fog_right_images).float()

        all_images = torch.cat([label_left_images, label_right_images, fog_left_images, fog_right_images], dim=0)
        if self.mode == 'train':
            croped_patches = random_crop_and_pad_image_and_labels(all_images, [self.im_height, self.im_width])
            fliped_patches = random_flip(croped_patches)
            all_patches = fliped_patches
        else:
            all_patches = all_images
        label_left_patches, label_right_patches, fog_left_patches, fog_right_patches = torch.split(all_patches, 7, dim=0)
        label_left_patches = label_left_patches.squeeze()
        label_right_patches = label_right_patches.squeeze()
        fog_left_patches = fog_left_patches.squeeze()
        fog_right_patches = fog_right_patches.squeeze()
        return (label_left_patches, label_right_patches, fog_left_patches, fog_right_patches)

if __name__ == '__main__':
    dataset = Trainval_Dataset(patch_size=256, fog_intensity=0.02, mode='val')
    validation_data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    with torch.no_grad():
        for iteration, batch in enumerate(validation_data_loader):
            a, b, c, d = batch[0], batch[1], batch[2], batch[3]
            print(a.shape)
            print(b.shape)
            print(c.shape)
            print(d.shape)
            