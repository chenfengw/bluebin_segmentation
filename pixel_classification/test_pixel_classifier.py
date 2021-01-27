'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


from __future__ import division
from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier
import numpy as np

if __name__ == '__main__':
  # test the classifier
  
  # folder = 'data/validation/blue'
  
  # X = read_pixels(folder)
  # myPixelClassifier = PixelClassifier()
  # y = myPixelClassifier.classify(X)
  
  # print('Precision: %f' % (sum(y==3)/y.shape[0]))

  folder = 'data/validation'
  X1 = read_pixels(folder+'/red', verbose = True)
  X2 = read_pixels(folder+'/green')
  X3 = read_pixels(folder+'/blue')
  y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
  X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))

  # save test data
  np.save("test_x.npy",X)
  np.save("test_y.npy",y)
  # classify
  myPixelClassifier = PixelClassifier()
  y_pred = myPixelClassifier.classify(X)
  print(sum(y_pred==y)/len(y))