# -----------
# Test with binary femur mask only
# -----------

# import all necessary packages
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import os
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from config import *

# config class
def predict(user_input):

    # load the trained weights into the model
    model = smp.Unet('resnet34', classes=2)
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    #model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    # create an example test set (this will be an user input later)
    # for now, using the patient 23's MRI scan as an example
    test_ex_dir = user_input # user input
    # if type(test_ex_dir) == 'ndarray':
    #     test_ex = test_ex_dir
    # else:
    #     test_ex = np.load(test_ex_dir)
    test_ex = test_ex_dir

    # Unstack all the 3D scan into ordered 2D scans
    list_2d = []
    if test_ex.shape[2] == None:
        list_2d = test_ex
    for i in range(test_ex.shape[2]):
        list_2d.append(test_ex[:,:,i])

    # test dataloader
    class CustomImageDataset(Dataset):
        def __init__(self, img_list):
            self.img_list = img_list

        def __len__(self):
            return len(self.img_list)

        def __getitem__(self, idx):
            tensor_image = self.img_list[idx]
            tensor_image = cv2.resize(tensor_image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
            image = np.zeros((3, tensor_image.shape[0], tensor_image.shape[1]))
            image[0, :,:] = tensor_image
            image[1, :,:] = tensor_image
            image[2, :,:] = tensor_image
            image = torch.Tensor(image/255)
            return image

    test_set = CustomImageDataset(img_list = list_2d)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Run predictions using the model
    pred_list = []
    model.eval()

    with torch.no_grad():
        for batch in test_dataloader:
            pred = model(batch)
            pred = torch.argmax(pred, 1).detach().cpu().squeeze()
            pred = pred.cpu().detach().numpy()
            pred_list.append(pred)

    pred_list = np.array(pred_list)
    np.save('test_data.npy', pred_list)
    return pred_list # 2D stack of prediction masks in a numpy array

    # Stack the 2D predictions slices and create a 3D structure -> bring in the MATLAB code

    # Export the 3D structure into .stl file

if __name__ == "__main__":
    test_ex_dir = 'data/images_3d/images_23.npy' # user input
    predict(test_ex_dir)