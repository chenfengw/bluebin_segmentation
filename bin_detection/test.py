#%%
import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import bin_detector
import  numpy as np
import importlib
from skimage.measure import label, regionprops, regionprops_table
from collections import defaultdict
importlib.reload(bin_detector)
# %%
%matplotlib inline

# %% test image segmentation
def get_all_imgs(data_dir):
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith("jpg")]
    return sorted(files)

# %% test bounding box
img_files_all = np.array(get_all_imgs("data/validation"))
clf = bin_detector.BinDetector()
images_all = defaultdict(list)

# img_files = img_files_all[[9,12,31]]
img_files = img_files_all
for img_file in img_files:
    img_bgr = cv2.imread(img_file)
    img_test = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_test = clf.segment_image(img_test)

    # calculate bounding box
    processed, bboxes = clf.get_bounding_boxes(mask_test,disk_size=3)

    # save all images for debug
    images_all["processed"].append(processed)
    images_all["rgb"].append(img_rgb)
    images_all["mask"].append(mask_test)
    images_all["hsv"].append(img_test)
    
    # show results
    plt.figure(figsize=(10,5))
    
    # rgb image
    plt.subplot(1,4,1)
    plt.title(img_file)
    plt.imshow(img_rgb)
    plt.axis("off")
    # hsv image
    plt.subplot(1,4,2)
    plt.imshow(img_test)
    plt.axis("off")

    # segmentation mask
    plt.subplot(1,4,3)
    plt.imshow(mask_test,cmap="gray")
    plt.axis("off")

    # show bounding box
    plt.subplot(1,4,4)
    plt.imshow(processed,cmap="gray")
    plt.title(len(bboxes))
    plt.axis("off")
    for _, bbox in bboxes:
        bx = bbox[0]
        by = bbox[1]
        plt.plot(bx, by, '-r', linewidth=2.5)
# %% test skimage
import skimage.morphology as morphology
image_indx = 0
label_img = label(images_all["mask"][image_indx])
selem = morphology.disk(5)
eroded = morphology.erosion(label_img, selem)
regions = regionprops(eroded)

minr, minc, maxr, maxc = regions[0].bbox
# bx = (minc, maxc, maxc, minc, minc)
# by = (minr, minr, maxr, maxr, minr)

bx = [minc, maxc, maxc, minc]
by = [minr, minr, maxr, maxr]

plt.figure()
plt.imshow(images_all["mask"][image_indx])

plt.figure()
plt.imshow(label_img)
plt.scatter(bx, by)
for i, (x, y) in enumerate(zip(bx,by)):
    plt.text(x,y,i,c="red")
# %% 
