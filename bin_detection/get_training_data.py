#%%
import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os 
import numpy as np
#%%
def get_all_imgs(data_dir):
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith("jpg")]
    return sorted(files)

#%% save image mask
all_imgs = get_all_imgs("data/training")
traning_masks = {}

for idx, img_str in enumerate(all_imgs):
    # load image
    img = cv2.imread(img_str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display the image and use roipoly for labeling
    plt.ion()
    plt.figure()
    plt.imshow(img)
    plt.title(img_str)
    my_roi = RoiPoly(color='r',close_fig=False,show_fig=True)
    # plt.close('all')

    # show mask
    mask = []
    mask = my_roi.get_mask(img[:,:,0])
    traning_masks[img_str] = mask
    print("saving img: {}".format(idx+1))
    plt.imshow(mask)

np.save("training_mask_likeblue.npy",traning_masks)

# %% 
def generate_training_samples(traning_masks,colorspace,mask_type="bluebins"):
    """Generate training samples from segmentation mask. 
    Separate image as bluebins and non blue bins

    Args:
        traning_masks (dict): keys: image file name, values segmask
        colorspace (str): color space to convert to. ie HLS, HSV, RGB
        mask_type (str, optional): bluebins or others. Defaults to "bluebins"
    Return:
        np array: training data, size n_pixels x 3
    """
    bin_pixels = []
    for idx, (img_str, img_mask) in enumerate(traning_masks.items()):
        # read image and convert color space
        img = cv2.imread(img_str)
        img = cv2.cvtColor(img, vars(cv2)["COLOR_BGR2" + colorspace])

        temp_piexel = []
        for i in range(img.shape[-1]):
            if mask_type == "bluebins":
                temp_piexel.append(img[img_mask,i])
            elif mask_type == "others":
                temp_piexel.append(img[~img_mask,i])
            elif mask_type == "likeblue":
                temp_piexel.append(img[img_mask,i])
        # blue_region size: n_pixels x 3
        blue_region = np.array(temp_piexel).T

        # collet data
        bin_pixels.append(blue_region)

    # save data
    bin_pixels_np = np.concatenate(bin_pixels)
    np.save("training_{}_{}.npy".format(mask_type,colorspace),bin_pixels_np)
    return bin_pixels_np

# %% generate_training_samples using seg mask
# colorspaces = ["HSV","HLS","LAB","RGB","YUV"]
colorspaces = ["HSV"]
# seg_types = ["bluebins","others"]
seg_types = ["likeblue"]

for colorspace in colorspaces:
    for seg_type in seg_types:
        print("{}-{}".format(colorspace,seg_type))
        generate_training_samples(traning_masks,colorspace,seg_type)

# %% create train x and train y
for color in colorspaces:
    train_bluebins = np.load("training_bluebins_{}.npy".format(color))
    train_others = np.load("training_others_{}.npy".format(color))
    # dist = np.linalg.norm(train_bluebins.mean(axis=0) - train_others.mean(axis=0))
    # print("{}, dist between mean is {}".format(color,dist))
