#-*- coding: utf-8 -*-
__version__ = '1.0.1'

import cv2
import numpy as np
import os
import svmutil

class SVM_Fault_Detector(object):
	def __init__(self, training_set_path, kernal_option = "-t 2"):
		self.kernal_option = kernal_option
		self.training_set_path = training_set_path
		self.model = self.svm_traing_process()

	def get_model(self):
		return self.model

	def compute_image_feature(self, im_gray):
		norm_im = cv2.resize(im_gray, (200,100), interpolation = cv2.INTER_CUBIC)
		norm_im = norm_im[3:-3,3:-3]
		return norm_im.flatten()

	def histeq(self, im,nbr_bins=256):
		""" Histogram equalization of a grayscale image. """
		# get image histogram
		imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
		cdf = imhist.cumsum() # cumulative distribution function
		cdf = 255 * cdf / cdf[-1] # normalize
		# use linear interpolation of cdf to find new pixel values
		im2 = np.interp(im.flatten(),bins[:-1],cdf)
		return im2.reshape(im.shape), cdf

	def load_images_dataset(self, path):
		imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(r".jpg")]
		labels = [int(imfile.split("/")[-1][-5]) for imfile in imlist]

		# create features from the images
		features = list()
		for imname in imlist:
			im_gray, cdf = self.histeq(cv2.imread(imname, 0))
			features.append(self.compute_image_feature(im_gray))

		return np.array(features), labels

	def svm_traing_process(self):
		features,labels = self.load_images_dataset(self.training_set_path)
		# train a SVM classifier
		features = map(list,features)
		prob = svmutil.svm_problem(labels,features)
		param = svmutil.svm_parameter(self.kernal_option)
		model =  svmutil.svm_train(prob,param)
		res = svmutil.svm_predict(labels,features,model)
		return model

	def svm_fault_detecting(self, imname):
		feature = list()
		im_gray, cdf = self.histeq(cv2.imread(imname, 0))
		feature.append(self.compute_image_feature(im_gray).tolist())

		p_label,p_acc,p_val = svmutil.svm_predict([0], feature, self.model)
		return p_label

def testing_process(fault_detector):

	test_features,test_labels = fault_detector.load_images_dataset(r"..//testing//")
	test_features = map(list,test_features)
	res = svmutil.svm_predict(test_labels,test_features,fd.get_model())

if __name__ == '__main__':
	fd = SVM_Fault_Detector(r"..//training//")
	print fd.svm_fault_detecting("..//testing//5_0.jpg")
	testing_process(fd)
