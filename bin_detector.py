'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
import skimage.morphology as morphology

class BinDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		self.mean = None
		self.cov = None
		self.prior = None

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

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image in BGR
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		if self.mean is None or self.cov is None:
			self.mean = np.load("weights/param_mean_HSV.npy",allow_pickle=True).item()
			self.cov = np.load("weights/param_cov_HSV.npy",allow_pickle=True).item()
			self.prior = np.load("weights/param_prior.npy",allow_pickle=True).item()

		# convert image to n_pixels x 3 
		img_new = np.reshape(img,[img.shape[0]*img.shape[1],img.shape[-1]])

		y_pred_all = []
		for class_label in ["others","bluebins","likeblue"]:
			y_pred = self.gaussian_pdf(img_new,self.mean[class_label],self.cov[class_label])
			y_pred_all.append(y_pred*self.prior[class_label])
		
		mask = np.argmax(y_pred_all,axis=0)
		mask = mask == 1 # 1-> bluebin
		return np.reshape(mask.astype(int), [img.shape[0],img.shape[1]])

	def get_bounding_boxes(self, mask_img, disk_size=3):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - masked img image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		mask_img = label(mask_img)
		selem = morphology.disk(disk_size)
		eroded = morphology.opening(mask_img, selem)
		regions = regionprops(eroded)

		# !debug: get all bounding boxes
		# box_all = []
		# for region in regions:
		# 	minr, minc, maxr, maxc = region.bbox
		# 	bx = (minc, maxc, maxc, minc, minc)
		# 	by = (minr, minr, maxr, maxr, minr)
		# 	box_all.append([bx,by])

		# get candidate
		box_candidates = []
		max_area = 0
		for region in regions:
			minr, minc, maxr, maxc = region.bbox
			width = maxc - minc
			height = maxr - minr
			area = region.bbox_area
			
			# reject area to small
			if  area < mask_img.shape[0]*mask_img.shape[1]/200:
				continue

			# reject long fat rectangle
			if width > 1.5*height:
				continue
			
			# get max area of connected components
			if area > max_area:
				max_area = area

			bx = (minc, maxc, maxc, minc, minc)
			by = (minr, minr, maxr, maxr, minr)
			box_candidates.append((area,[bx,by]))
		
		# make another slection based on area ratio
		box_final = []
		for box_area, box_coord in box_candidates:
			if box_area < 0.1*max_area:
				continue
			box_final.append(box_coord)

		# return top left, bottom right
		# top left: (minc,minr) bottom right: (maxc,maxr)
		box_coord = [[box[0][0], box[1][0], box[0][1], box[1][2]] for box in box_final]
		# return eroded, box_candidates
		return box_coord

