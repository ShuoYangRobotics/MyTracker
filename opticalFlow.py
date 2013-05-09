#!/usr/bin/env python
# encoding: utf-8
"""
opticalFlow.py

Created by Yang Shuo on 2013-05-07.
Copyright (c) 2013 . All rights reserved.

This code use SIFT + optical flow to do tracking.
It prerequisites a bounding box. 
First, SIFT of the entire image is extracted.
Second, optical flow of adjcent frames are calculated.
"""

import sys
import os     
import cv      
import cv2
from numpy import * 
from pylab import *
from PIL import Image
from time import *   
from homography import *
import sift

def mynorm(point1,point2):
	return sqrt((point1[0] - point2[0])*(point1[0] - point2[0])+(point1[1] - point2[1])*(point1[1] - point2[1]))

def windowWeight(point, ver1, ver2):
	"""
		2013-05-09
		This function takes in a point and two boundary points of a rectangle
		depends on the relative position of the point to the rectangle
		different weight is given  
	"""
	frac = 0.15                            
	wf = (ver2[0] - ver1[0])*frac
	hf = (ver2[1] - ver1[1])*frac
	if point[0]>= ver1[0]+wf and point[0]<= ver2[0]-wf:
		if point[1]>= ver1[1]+hf and point[1]<= ver2[1]-hf: 
			return 1
		elif point[1]>= ver1[1] and point[1]<= ver1[1]+hf:
			return (point[1] - ver1[1])/hf
		elif point[1]>= ver2[1]-hf and point[1]<= ver2[1]:
			return (ver2[1] - point[1])/hf	
		else:			                 
			return 0.01
	elif point[0]>= ver1[0] and point[0]<= ver1[0]+wf:
		if point[1]>= ver1[1]+hf and point[1]<= ver2[1]-hf: 
			return (point[0] - ver1[0])/wf
		elif point[1]>= ver1[1] and point[1]<= ver1[1]+hf:
			return 0.02
		elif point[1]>= ver2[1]-hf and point[1]<= ver2[1]:
			return 0.02	
		else:			                 
			return 0.01 		
	elif point[0]>= ver2[0]-wf and point[0]<= ver2[0]:
		if point[1]>= ver1[1]+hf and point[1]<= ver2[1]-hf: 
			return (ver2[0]-point[0])/wf
		elif point[1]>= ver1[1] and point[1]<= ver1[1]+hf:
			return 0.02
		elif point[1]>= ver2[1]-hf and point[1]<= ver2[1]:
			return 0.02	
		else:
			return 0.01
	else:
		return 0.005 
                           			  
def main():                                                   
	path = "../rawData/outdoor1/GOPR0003_0/"
	files = os.listdir(path)  
	"""
		Get SIFT features for the first frame
	"""
	imname = path+files[10]
	a =  time()
	sift.process_image(imname,'empire.sift')          
	l1,d1 = sift.read_features_from_file('empire.sift')   
	b = time()
	print "Time for getting SIFT features: ",b - a
	
	refine_l = []  
	""" define the bounding box and filter the points"""
	b_ver1 = (60,91)
	b_ver2 = (148,315)  
	w = (b_ver2[0] - b_ver1[0]) 
	h = (b_ver2[1] - b_ver1[1]) 
	initalIndicator = 1        #indicate the scale of the box
	for i,p in enumerate(l1):
		center = p[:2] 
		if center[0] > 60 and center[0] < 148 and center[1] > 91 and center[1] < 315:
			refine_l.append(tuple(center))
			
	color = (0,0,255)
	""" start at frame285 """
	startIndex = 10
	#writer = cv.CreateVideoWriter("myTrack.avi",cv.CV_FOURCC('M','J','P','G'),60,cv.GetSize (cv.LoadImage(path+files[startIndex])),1)
   
	while (startIndex<=2900):
		frameStartTime =  time()
		seq = [cv.LoadImage(path+files[startIndex]),
			cv.LoadImage(path+files[startIndex+1])]
		"""
			calculate optical flow
		"""                
		newFrameImageGS_32F1 = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		newFrameImageGS_32F2 = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		cv.CvtColor(seq[0],newFrameImageGS_32F1,cv.CV_RGB2GRAY)
		cv.CvtColor(seq[1],newFrameImageGS_32F2,cv.CV_RGB2GRAY)
		pyramid = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		prev_pyramid = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		flags = 0 
		     
		points, status1, errors,= cv.CalcOpticalFlowPyrLK (
		                    newFrameImageGS_32F1, 
		                    newFrameImageGS_32F2, 
		                    prev_pyramid, 
		                    pyramid,
		                    refine_l,
		                    (10,10),
		                    3,#pyr number
		                    (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 10, 0.01),
		                    flags)
		
		"""center of mass of points"""
		p_x = 0.0
		p_y = 0.0
		sumWeight = 0.0
		for i,point in enumerate(points):
			myweight = windowWeight(point,b_ver1,b_ver2)
			p_x += myweight*point[0]
			p_y += myweight*point[1]
			sumWeight += myweight
		# p_x = [windowWeight(point,b_ver1,b_ver2)*point[0] for point in points]
		# p_y = [windowWeight(point,b_ver1,b_ver2)*point[1] for point in points]		   
		# weight = [windowWeight(point,b_ver1,b_ver2) for point in points]
		cm_x = p_x/sumWeight
		cm_y = p_y/sumWeight
		
		""" remove far away points """ 
		diag = mynorm(b_ver1,b_ver2)
		points = [point for point in points if mynorm(point,(cm_x,cm_y)) <= 0.7*diag]
			  	  	
		"""
			detect scale change by finding the density of points
		"""                                                     
		densityIndicator = sum([mynorm(point,(cm_x,cm_y)) for point in points])/len(points)
		if startIndex == 10:
			initalIndicator = densityIndicator
			
		#print "Scale indicator ", densityIndicator/initalIndicator  
 
		""" Calculate new bounding box """
		new_w = w*densityIndicator/initalIndicator
		new_h = h*densityIndicator/initalIndicator
		b_ver1 = (int(cm_x-new_w/2), int(cm_y-new_h/2))
		b_ver2 = (int(cm_x+new_w/2), int(cm_y+new_h/2))

		frameEndTime = time()
  	    
		print "Time to process one frame: ", frameEndTime - frameStartTime
		
		""" draw images and detected points"""
		for pt in refine_l:
			cv.Circle(seq[0], (int(pt[0]), int(pt[1])), 5, color, 0, cv.CV_AA, 0)
		cv.Circle(seq[0], (int(cm_x), int(cm_y)), 10, (0,255,255), 0, cv.CV_AA, 0)	 
		cv.Rectangle(seq[0],b_ver1,b_ver2,(255,0,0))
		cv.ShowImage('First', seq[0])   
		cv.WaitKey(30)
		for pt in points:
			cv.Circle(seq[1], (int(pt[0]), int(pt[1])), 5, color, 0, cv.CV_AA, 0)
		cv.Circle(seq[1], (int(cm_x), int(cm_y)), 10, (0,255,255), 0, cv.CV_AA, 0)	
		cv.Rectangle(seq[1],b_ver1,b_ver2,(255,0,0))
		cv.ShowImage('First', seq[1])

		refine_l = points
		startIndex += 1
		#cv.WriteFrame(writer, seq[0])
		#cv.WriteFrame(writer, seq[1])

if __name__ == '__main__':
	main()

