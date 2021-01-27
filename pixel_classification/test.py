#%% 
import numpy as np
import pixel_classifier 
import importlib
 
importlib.reload(pixel_classifier)
#%% test pixel classiefer classify
clf = pixel_classifier.PixelClassifier()
train_x, train_y = np.load("train_x.npy"), np.load("train_y.npy")

# %%
y_pred = clf.classify(train_x)
y_pred_new = np.argmax(y_pred,axis=0) + 1
# %%
# try testing data
test_x, test_y = np.load("test_x.npy"), np.load("test_y.npy")
# %%
