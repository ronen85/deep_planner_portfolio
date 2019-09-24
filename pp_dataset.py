import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import io, transform, util
from scipy.fftpack import fft2
from sklearn.preprocessing import RobustScaler, MinMaxScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlannerPortfolioDataset(Dataset):
    """

    """

    def __init__(self, csv_file, image_dir, ftransform=None):
        """
        Args:
         csv_file (string): a path to a csv file that describe which planner is preferable for each problem.
         image_dir (string): a path to a directory with all the images that describe the planning problem
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.ftransform = ftransform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        planner_results = np.array(self.df.iloc[idx, 1:])
        planner_results = planner_results.astype('float32')
        planner_results = torch.from_numpy(planner_results)
        planner_results = planner_results.to(device)

        problem_name = self.df.iloc[idx]['filename']
        problem_image_path = os.path.join(self.image_dir, problem_name + '-bolded-cs.png')

        image = io.imread(problem_image_path)
        image = util.invert(image)
        image = np.asarray(image)
        image = image.astype('float32')
        image = image / 255.0  # Normalize the data

        if self.ftransform:
            transformed_image = fft2(image)
            transformed_image = transformed_image / transformed_image[0, 0]
            real_transformed_image = np.real(transformed_image)
            img_transformed_image = np.imag(transformed_image)
            image = np.dstack((real_transformed_image, img_transformed_image))
            image = np.swapaxes(image, 0, -1)
            image = torch.from_numpy(image)
            image = image.to(device)
            # image.unsqueeze_(0)
        else:
            image = torch.from_numpy(image)
            image = image.to(device)
            image.unsqueeze_(0)

        return image, planner_results