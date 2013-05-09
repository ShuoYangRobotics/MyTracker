#!/usr/bin/env python
# encoding: utf-8
"""
HOG.py

Created by Yang Shuo on 2013-05-07.
Copyright (c) 2013 . All rights reserved.
"""

import sys
import os     
import cv      
from numpy import * 
from time import *   
from homography import *

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
	#figure()
	#gray()   
	b_ver1 = (60,91)
	b_ver2 = (148,315) 
	for i,p in enumerate(l1):
		center = p[:2] 
		if i % 30 ==0:
			# if center[0] < 60 or center[0] > 148:
			# 	continue
			# if center[1] < 91 or center[1] > 315:
			# 	continue  
			refine_l.append(tuple(center))	
			# plot(center[0],center[1],'ob')  
			# 		imshow(im1)                    
			# 				axis('off')
		elif center[0] > 60 and center[0] < 148 and center[1] > 91 and center[1] < 315:
			refine_l.append(tuple(center))
	#sift.plot_features(im1,l1,circle=True)
	#show()
	
	startIndex = 10
	
	while (startIndex<=90):
		seq = [cv.LoadImage(path+files[startIndex]),
			cv.LoadImage(path+files[startIndex+1])]  
		#draw a box on image 1
		# cv.Rectangle(seq[0],(60,91),(148,315),(255,0,0))   
		# 	cv.ShowImage('TestOpticFlow', seq[0])   
		# 	
		# 	cv.WaitKey()
		# 	cv.DestroyAllWindows()                   
	
		"""
			calculate optical flow
		"""  
		color = (0,0,255)             
		newFrameImageGS_32F1 = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		newFrameImageGS_32F2 = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		cv.CvtColor(seq[0],newFrameImageGS_32F1,cv.CV_RGB2GRAY)
		cv.CvtColor(seq[1],newFrameImageGS_32F2,cv.CV_RGB2GRAY)
		pyramid = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		prev_pyramid = cv.CreateImage (cv.GetSize (seq[0]), 8, 1)
		flags = 0 
		#a =  time()     
	
		points, status, errors,= cv.CalcOpticalFlowPyrLK (
		                    newFrameImageGS_32F1, 
		                    newFrameImageGS_32F2, 
		                    prev_pyramid, 
		                    pyramid,
		                    refine_l,
		                    (10,10),
		                    6,#pyr number
		                    (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 10, 0.01),
		                    flags)
		#b = time()
		#print "Time for calculating optical flow: ",b - a
	   
		#a =  time()
		fp = make_homog(transpose(array(refine_l)))
		tp = make_homog(transpose(array(points)))
		H = H_from_points(fp,tp) 
		#H = Haffine_from_points(fp,tp)
		#b = time()
		#print "Time for calculating homography: ",b - a  
		#print H[0:2,0:2]  
		H_solve_s_and_ang(H[0:2,0:2])
		#print H[0:2,2]
		disp = (int(H[0:2,2][0]),int(H[0:2,2][1]))
	
		cv.Rectangle(seq[0],b_ver1,b_ver2,(255,0,0))
		for pt in refine_l:
			cv.Circle(seq[0], (int(pt[0]), int(pt[1])), 5, color, 0, cv.CV_AA, 0)
		cv.ShowImage('First', seq[0])   
		cv.WaitKey(30)
		
		b_ver1 = (b_ver1[0]+disp[0],b_ver1[1]+disp[1])
		b_ver2 = (b_ver2[0]+disp[0],b_ver2[1]+disp[1])
		
		cv.Rectangle(seq[1],b_ver1,b_ver2,(255,0,0)) 
		#cv.DestroyWindow('First')
		for pt in points:
			cv.Circle(seq[1], (int(pt[0]), int(pt[1])), 5, color, 0, cv.CV_AA, 0)
		cv.ShowImage('First', seq[1])
		#sleep(0.7)  
		cv.WaitKey(5)
		#cv.DestroyWindow('First')    
		
		
		startIndex += 1
	


if __name__ == '__main__':
	main()

