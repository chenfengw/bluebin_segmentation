'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.mean = None
    self.cov = None
    self.n_class = 0
    self.class_idx = None
  def train(self,X,y):
    """calculate parameter of model

    Args:
        X (training samples): shape: n_sample x 3 (rgb)
        y (labels): shape: n_samples x 1, row vector
    """
    self.mean = {}
    self.cov = {}
    self.class_idx = sorted(set(y))
    self.n_class = len(self.class_idx)

    for color_idx in self.class_idx:
      # calculate parameter for gaussian
      samples_x = X[y==color_idx]
      self.mean[color_idx] = np.mean(samples_x,axis=0)
      self.cov[color_idx] = np.cov(samples_x,rowvar=False)
      assert self.mean[color_idx].shape[0] == self.cov[color_idx].shape[0]

  def gaussian_pdf(self,X,mean,cov):
    """Compute multivariable gaussian pdf

    Args:
        mean (np array): n_dim x 1, row vector
        cov (np array): n_dim x n_dim

    Returns:
        np array: value of gaussian pdf, row vector
    """
    assert mean.shape[0] == cov.shape[0],"dim must match"
    n_dim = len(mean)
    x_new = X - mean

    # calculate the mahalanobis distance
    mh_dist = np.einsum('ij,ij->i', x_new @ np.linalg.inv(cov), x_new)
    reg = 1/np.sqrt((2*np.pi)**n_dim * np.linalg.det(cov)) 
    return reg * np.exp(-0.5 * mh_dist)

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: 1 x n vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    # YOUR CODE HERE
    # Just a random classifier for now
    # Replace this with your own approach 
    # y = 1 + np.random.randint(3, size=X.shape[0])
    # print(y.shape)

    # training 
    x_train = np.load("train_x.npy")
    y_train = np.load("train_y.npy")
    self.train(x_train,y_train)

    y_pred_all = []
    for label in self.class_idx:
      y_pred = self.gaussian_pdf(X,self.mean[label],self.cov[label])
      y_pred_all.append(y_pred)
      
    return np.argmax(y_pred_all,axis=0) + 1

