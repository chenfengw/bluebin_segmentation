#%%
import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os 
import numpy as np

# %% train data
def train_gaussian_classifer(colorspace="HSV"):
    param_mean = {}
    param_cov = {}
    param_prior = {}
    total_samples = 0

    for class_type in ["bluebins","others",'likeblue']:
        data = np.load("data/training/training_{}_{}.npy".format(class_type,colorspace))
        param_mean[class_type] = np.mean(data,axis=0)
        param_cov[class_type] = np.cov(data,rowvar=False)
        param_prior[class_type] = len(data) # class frequency
        total_samples += len(data)

    # calculate class frequency
    param_prior = {key : val/total_samples for key,val in param_prior.items()}
    
    # save data 
    np.save("param_mean_{}".format(colorspace),param_mean)
    np.save("param_cov_{}".format(colorspace),param_cov)
    np.save("param_prior",param_prior)
# %% perform training
if __name__ == "__main__":
    train_gaussian_classifer(colorspace="HSV")

# %%
